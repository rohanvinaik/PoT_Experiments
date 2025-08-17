# Attack and Defense Testing Framework

Comprehensive testing suite for the PoT attack and defense mechanisms.

## Test Structure

```
tests/test_attacks/
├── test_attack_suite.py      # Attack suite functionality tests
├── test_defenses.py          # Defense mechanism tests
├── test_integration.py       # End-to-end integration tests
├── test_core_functionality.py # Core component tests
└── run_all_tests.py         # Test runner script
```

## Test Coverage

### Attack Testing (`test_attack_suite.py`)
- **Distillation Attacks**: Weak, moderate, strong configurations
- **Compression Attacks**: Pruning and quantization variants
- **Wrapper Detection**: Statistical anomaly detection
- **Adaptive Attacks**: Evolution and defense observation
- **Vision Attacks**: Adversarial patches, universal perturbations, model extraction, backdoors
- **Attack Metrics**: Success rates, robustness scoring, performance degradation

### Defense Testing (`test_defenses.py`)
- **Defense Effectiveness**: Against various attack types
- **Adaptive Thresholds**: Dynamic threshold adjustment
- **Input Sanitization**: Filtering and outlier removal
- **Randomized Defenses**: Smoothing and stochastic verification
- **Integrated Defense**: Full pipeline testing
- **Performance Metrics**: False positive rates, detection accuracy, overhead

### Integration Testing (`test_integration.py`)
- **End-to-End Pipeline**: Attack → Detection → Defense → Verification
- **Attack-Defense Arms Race**: Adaptive evolution testing
- **Multi-Stage Attacks**: Layered attack and defense responses
- **Vision Integration**: Vision-specific attack and defense
- **Performance Monitoring**: Defense adaptation over time

### Core Functionality (`test_core_functionality.py`)
- **Configuration Creation**: Attack and defense configs
- **Component Initialization**: All classes and modules
- **API Verification**: Method existence and signatures
- **Import Testing**: Module dependencies

## Running Tests

### Run All Tests
```bash
python tests/test_attacks/run_all_tests.py
```

### Run Specific Test Suite
```bash
# Attack tests
python tests/test_attacks/test_attack_suite.py

# Defense tests  
python tests/test_attacks/test_defenses.py

# Integration tests
python tests/test_attacks/test_integration.py

# Core functionality
python tests/test_attacks/test_core_functionality.py
```

### Quick Integration Test
```bash
python test_attack_defense_integration.py
```

## Test Classes

### Attack Testing Classes
- `TestDistillationAttack`: Distillation attack variants
- `TestCompressionAttack`: Compression techniques
- `TestWrapperDetection`: Wrapper detection accuracy
- `TestAdaptiveAttacks`: Adaptive attack generation
- `TestVisionAttacks`: Vision-specific attacks
- `TestAttackIntegration`: Attack pipeline integration
- `TestAttackMetrics`: Attack evaluation metrics

### Defense Testing Classes
- `TestDefenseEffectiveness`: Defense against attacks
- `TestAdaptiveThreshold`: Threshold adaptation
- `TestInputSanitization`: Input filtering
- `TestRandomizedDefense`: Randomization techniques
- `TestIntegratedDefense`: Complete defense system
- `TestDefenseMetrics`: Defense evaluation

### Integration Testing Classes
- `TestEndToEndPipeline`: Full attack-defense cycle
- `TestAttackRunnerPipeline`: Attack evaluation harness
- `TestVisionAttackIntegration`: Vision attack integration
- `TestDefenseAdaptation`: Defense learning

## Key Test Scenarios

### 1. Basic Attack-Defense
```python
# Create attack
attack_config = AttackConfig(
    name="test",
    attack_type="distillation",
    budget={'queries': 1000},
    strength='moderate',
    success_metrics={'accuracy_drop': 0.1}
)

# Run attack
runner = AttackRunner()
result = runner.run_single_attack(model, attack_config, data_loader)

# Apply defense
defense = IntegratedDefenseSystem(base_verifier)
defense_result = defense.comprehensive_defense(input_data, model)
```

### 2. Adaptive Evolution
```python
# Adaptive attack suite
adaptive_suite = AdaptiveAttackSuite(population_size=10)
evolved_attacks = adaptive_suite.evolve_attacks(model, data_loader)

# Adaptive defense
adaptive_verifier = AdaptiveVerifier(base_verifier)
adaptive_verifier.learn_from_attacks(evolved_attacks)
```

### 3. Vision-Specific
```python
# Vision attack
result = execute_vision_attack(
    attack_type='adversarial_patch',
    config={'patch_size': 8},
    model=model,
    data_loader=loader
)

# Vision defense
defense.comprehensive_defense(patched_images, model)
```

## Test Metrics

### Attack Metrics
- **Success Rate**: Percentage of successful attacks
- **Confidence**: Attack confidence scores
- **Robustness Score**: Model resistance to attacks
- **Performance Degradation**: Accuracy/latency impact

### Defense Metrics
- **Detection Rate**: Attack detection accuracy
- **False Positive Rate**: Clean data flagged as attacks
- **Defense Overhead**: Computational cost
- **Adaptation Rate**: Learning from attacks

## Mock Components

### MockBaseVerifier
Simple verifier for testing defense systems without full PoT implementation.

### Mock Attack Results
Simulated attack outcomes for testing defense responses.

### Mock Data
Synthetic data generators for consistent testing.

## Continuous Testing

### Pre-commit Checks
```bash
# Run core tests before committing
python tests/test_attacks/test_core_functionality.py
```

### Full Test Suite
```bash
# Complete test suite for thorough validation
bash run_all_tests.sh
```

## Known Issues and Limitations

1. **Parallel Testing**: Currently uses sequential execution for simplicity
2. **GPU Testing**: Most tests run on CPU for compatibility
3. **Large-Scale Testing**: Limited to small models/datasets for speed
4. **Stochastic Tests**: Some randomness in adaptive components

## Future Enhancements

- [ ] Performance benchmarking suite
- [ ] Stress testing with large models
- [ ] GPU-accelerated test variants
- [ ] Coverage reporting integration
- [ ] Automated regression testing
- [ ] Attack effectiveness benchmarks

## Contributing

When adding new attacks or defenses:
1. Add corresponding test cases
2. Update integration tests
3. Document test scenarios
4. Ensure all tests pass before PR