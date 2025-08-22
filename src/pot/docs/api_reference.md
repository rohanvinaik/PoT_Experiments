# API Reference - Attack Resistance Module

## Core Components

### pot.core.attack_suites

#### Classes

##### `AttackConfig`
```python
@dataclass
class AttackConfig:
    """
    Configuration for an attack.
    
    Attributes:
        name (str): Unique identifier for the attack
        attack_type (str): Type of attack (distillation, compression, wrapper, etc.)
        budget (Dict[str, Any]): Resource constraints (queries, time, memory)
        strength (str): Attack strength level (weak, moderate, strong, adaptive)
        success_metrics (Dict[str, float]): Target metrics for success evaluation
        parameters (Dict[str, Any]): Attack-specific parameters
        metadata (Dict[str, Any]): Additional metadata
    
    Methods:
        scale_budget(factor: float) -> AttackConfig:
            Create a new config with scaled budget
        
        validate() -> bool:
            Validate configuration parameters
    
    Example:
        >>> config = AttackConfig(
        ...     name="distillation_test",
        ...     attack_type="distillation",
        ...     budget={'queries': 1000},
        ...     strength='moderate',
        ...     success_metrics={'accuracy_drop': 0.1}
        ... )
    """
```

##### `StandardAttackSuite`
```python
class StandardAttackSuite:
    """
    Standard collection of attacks for benchmarking.
    
    Methods:
        get_distillation_configs() -> List[AttackConfig]:
            Get predefined distillation attack configurations
            
        get_compression_configs() -> List[AttackConfig]:
            Get predefined compression attack configurations
            
        get_wrapper_configs() -> List[AttackConfig]:
            Get predefined wrapper attack configurations
            
        get_vision_specific_configs() -> List[AttackConfig]:
            Get vision-specific attack configurations
            
        execute_attack(config: AttackConfig, model: nn.Module, 
                      data_loader: DataLoader) -> Dict[str, Any]:
            Execute a single attack
    
    Example:
        >>> suite = StandardAttackSuite()
        >>> configs = suite.get_distillation_configs()
        >>> result = suite.execute_attack(configs[0], model, data)
    """
```

##### `AdaptiveAttackSuite`
```python
class AdaptiveAttackSuite:
    """
    Adaptive attack suite with evolutionary algorithms.
    
    Parameters:
        population_size (int): Size of attack population
        mutation_rate (float): Mutation probability
        crossover_rate (float): Crossover probability
        
    Methods:
        evolve_attacks(model: nn.Module, data_loader: DataLoader,
                      generations: int = 10) -> List[AttackConfig]:
            Evolve attack configurations
            
        observe_defense(defense_result: Dict) -> None:
            Learn from defense responses
            
        generate_evasive_attack(base_config: AttackConfig) -> AttackConfig:
            Generate attack to evade observed defenses
    
    Example:
        >>> adaptive = AdaptiveAttackSuite(population_size=20)
        >>> evolved = adaptive.evolve_attacks(model, data, generations=5)
    """
```

##### `AttackRunner`
```python
class AttackRunner:
    """
    Execute and evaluate attack suites.
    
    Parameters:
        device (str): Device to run attacks on
        verbose (bool): Print progress information
        
    Methods:
        run_single_attack(model: nn.Module, config: AttackConfig,
                         data_loader: DataLoader) -> Dict[str, Any]:
            Execute a single attack
            
        run_attack_suite(model: nn.Module, configs: List[AttackConfig],
                        data_loader: DataLoader) -> List[Dict]:
            Run multiple attacks sequentially
            
        calculate_metrics(results: List[Dict]) -> Dict[str, Any]:
            Calculate aggregate metrics
            
        calculate_robustness_score(results: List[Dict]) -> float:
            Compute overall robustness score (0-1)
    
    Example:
        >>> runner = AttackRunner(device='cuda')
        >>> results = runner.run_attack_suite(model, configs, data)
        >>> score = runner.calculate_robustness_score(results)
    """
```

### pot.core.defenses

#### Classes

##### `DefenseConfig`
```python
@dataclass
class DefenseConfig:
    """
    Configuration for defense mechanisms.
    
    Attributes:
        adaptive_threshold (float): Initial adaptive threshold
        input_filter_strength (float): Filter strength (0-1)
        randomization_scale (float): Noise scale for randomization
        ensemble_size (int): Number of models in ensemble
        update_frequency (int): Threshold update frequency
        
    Methods:
        to_dict() -> Dict[str, Any]:
            Convert to dictionary
            
        from_yaml(path: str) -> DefenseConfig:
            Load from YAML file
    
    Example:
        >>> config = DefenseConfig(
        ...     adaptive_threshold=0.05,
        ...     input_filter_strength=0.7,
        ...     randomization_scale=0.1
        ... )
    """
```

##### `AdaptiveVerifier`
```python
class AdaptiveVerifier:
    """
    Verifier with adaptive threshold adjustment.
    
    Parameters:
        base_verifier: Base verification system
        config: Defense configuration
        
    Methods:
        verify(model: nn.Module, input_data: torch.Tensor,
               reference_output: torch.Tensor) -> Dict[str, Any]:
            Perform adaptive verification
            
        update_from_attack(attack_result: Dict) -> None:
            Update thresholds based on attack
            
        learn_attack_pattern(attack_info: Dict) -> None:
            Learn attack patterns for detection
            
        get_attack_patterns() -> Dict[str, Dict]:
            Get learned attack patterns
    
    Properties:
        threshold (float): Current adaptive threshold
        attack_history (List[Dict]): History of observed attacks
        
    Example:
        >>> adaptive = AdaptiveVerifier(base_verifier, config)
        >>> result = adaptive.verify(model, input_data, reference)
        >>> adaptive.update_from_attack({'success': True})
    """
```

##### `InputFilter`
```python
class InputFilter:
    """
    Filter and sanitize potentially adversarial inputs.
    
    Parameters:
        config: Filter configuration
        
    Methods:
        detect_adversarial(input_data: torch.Tensor,
                          reference_dist: Dict = None) -> Dict[str, Any]:
            Detect adversarial inputs
            
        sanitize(input_data: torch.Tensor, threat_level: float = 0.5,
                apply_blur: bool = True, apply_median: bool = True,
                apply_quantization: bool = True) -> Dict[str, Any]:
            Sanitize input with multiple filters
            
        detect_outliers(input_data: torch.Tensor,
                       threshold: float = 3.0) -> Dict[str, Any]:
            Detect statistical outliers
    
    Example:
        >>> filter = InputFilter(config)
        >>> detection = filter.detect_adversarial(input_data)
        >>> if detection['is_adversarial']:
        ...     result = filter.sanitize(input_data)
        ...     clean_input = result['sanitized_input']
    """
```

##### `RandomizedDefense`
```python
class RandomizedDefense:
    """
    Defense using randomization techniques.
    
    Parameters:
        config: Randomization configuration
        
    Methods:
        random_smoothing(model: nn.Module, input_data: torch.Tensor,
                        num_samples: int = 100, 
                        noise_scale: float = 0.1) -> Dict[str, Any]:
            Apply randomized smoothing
            
        stochastic_verify(model: nn.Module, input_data: torch.Tensor,
                         reference: torch.Tensor,
                         num_trials: int = 20) -> Dict[str, Any]:
            Stochastic verification with multiple trials
            
        compute_certified_radius(base_pred: torch.Tensor,
                                smooth_preds: List[torch.Tensor],
                                noise_scale: float,
                                confidence: float = 0.95) -> float:
            Compute certified robust radius
    
    Example:
        >>> defense = RandomizedDefense(config)
        >>> smooth = defense.random_smoothing(model, input_data)
        >>> radius = defense.compute_certified_radius(
        ...     base_pred, smooth_preds, 0.1
        ... )
    """
```

##### `IntegratedDefenseSystem`
```python
class IntegratedDefenseSystem:
    """
    Comprehensive defense system integrating multiple mechanisms.
    
    Parameters:
        base_verifier: Base verification system
        defense_configs: Configuration for each defense component
        
    Methods:
        comprehensive_defense(input_data: torch.Tensor,
                            model: nn.Module,
                            threat_level: float = None) -> Dict[str, Any]:
            Apply full defense pipeline
            
        get_performance_metrics() -> Dict[str, Any]:
            Get system performance metrics
            
        calculate_false_positive_rate(results: List[Dict]) -> float:
            Calculate false positive rate
            
        calculate_detection_accuracy(results: List[Dict]) -> float:
            Calculate detection accuracy
    
    Example:
        >>> defense = IntegratedDefenseSystem(verifier, configs)
        >>> result = defense.comprehensive_defense(
        ...     input_data, model, threat_level=0.7
        ... )
        >>> print(f"Verified: {result['final_decision']['verified']}")
    """
```

### pot.vision.attacks

#### Classes

##### `AdversarialPatchAttack`
```python
class AdversarialPatchAttack:
    """
    Generate adversarial patches for vision models.
    
    Parameters:
        patch_size (int): Size of the patch
        optimization_steps (int): Number of optimization steps
        optimizer_type (str): Optimizer to use (adam, sgd, momentum)
        
    Methods:
        generate_patch(model: nn.Module, data_loader: DataLoader,
                      target_class: int, device: str = 'cpu') -> torch.Tensor:
            Generate adversarial patch
            
        apply_patch(images: torch.Tensor, patch: torch.Tensor,
                   location: Tuple[int, int] = None) -> torch.Tensor:
            Apply patch to images
            
        evaluate_patch(patch: torch.Tensor, model: nn.Module,
                      data_loader: DataLoader) -> Dict[str, float]:
            Evaluate patch effectiveness
    
    Example:
        >>> attack = AdversarialPatchAttack(patch_size=32)
        >>> patch = attack.generate_patch(model, data, target_class=0)
        >>> patched_images = attack.apply_patch(images, patch)
    """
```

##### `UniversalPerturbationAttack`
```python
class UniversalPerturbationAttack:
    """
    Generate universal adversarial perturbations.
    
    Parameters:
        epsilon (float): Maximum perturbation magnitude
        max_iterations (int): Maximum iterations
        overshoot (float): Overshoot parameter
        
    Methods:
        generate_universal_perturbation(model: nn.Module,
                                      data_loader: DataLoader,
                                      device: str = 'cpu') -> torch.Tensor:
            Generate universal perturbation
            
        evaluate_transferability(perturbation: torch.Tensor,
                               source_model: nn.Module,
                               target_model: nn.Module,
                               data_loader: DataLoader) -> float:
            Evaluate cross-model transferability
    
    Example:
        >>> attack = UniversalPerturbationAttack(epsilon=0.05)
        >>> perturbation = attack.generate_universal_perturbation(
        ...     model, data_loader
        ... )
    """
```

##### `execute_vision_attack`
```python
def execute_vision_attack(attack_type: str,
                         config: Dict[str, Any],
                         model: nn.Module,
                         data_loader: DataLoader,
                         device: str = 'cpu') -> AttackResult:
    """
    Execute vision-specific attack with unified interface.
    
    Parameters:
        attack_type: Type of attack to execute
        config: Attack configuration parameters
        model: Target model
        data_loader: Data for attack
        device: Device to run on
        
    Returns:
        AttackResult with success status and metadata
        
    Supported attack types:
        - adversarial_patch
        - universal_perturbation
        - model_extraction
        - backdoor
        
    Example:
        >>> result = execute_vision_attack(
        ...     'adversarial_patch',
        ...     {'patch_size': 32, 'epsilon': 0.03},
        ...     model, data_loader
        ... )
    """
```

### pot.eval.attack_benchmarks

#### Classes

##### `AttackBenchmark`
```python
class AttackBenchmark:
    """
    Standardized benchmark for attack resistance.
    
    Parameters:
        device (str): Device for benchmarking
        verbose (bool): Print progress
        save_results (bool): Save results to disk
        results_dir (str): Directory for results
        
    Class Attributes:
        STANDARD_ATTACKS (List[str]): List of standard attack names
        
    Methods:
        run_benchmark(model: nn.Module, verifier: Any,
                     data_loader: DataLoader,
                     attack_names: List[str] = None,
                     include_defenses: bool = True) -> pd.DataFrame:
            Run benchmark suite
            
        compute_robustness_score(results: pd.DataFrame) -> float:
            Compute robustness score (0-100)
            
        generate_leaderboard(results_dict: Dict[str, pd.DataFrame],
                           save_path: str = None) -> pd.DataFrame:
            Generate model comparison leaderboard
            
        generate_report(results: pd.DataFrame,
                       save_path: str = None) -> Dict[str, Any]:
            Generate comprehensive report
    
    Example:
        >>> benchmark = AttackBenchmark(device='cuda')
        >>> results = benchmark.run_benchmark(model, verifier, data)
        >>> score = benchmark.compute_robustness_score(results)
        >>> print(f"Robustness: {score:.1f}/100")
    """
```

##### `AttackMetricsDashboard`
```python
class AttackMetricsDashboard:
    """
    Interactive dashboard for attack metrics.
    
    Parameters:
        results_dir (str): Directory containing benchmark results
        
    Methods:
        create_dashboard(output_path: str = None) -> None:
            Create full interactive dashboard
            
        plot_attack_success_rates(save_path: str = None) -> go.Figure:
            Plot success rates by attack type
            
        plot_far_frr_tradeoffs(save_path: str = None) -> go.Figure:
            Plot FAR/FRR trade-off curves
            
        plot_defense_adaptation(save_path: str = None) -> go.Figure:
            Plot defense adaptation over time
            
        generate_summary_statistics() -> pd.DataFrame:
            Generate summary statistics table
    
    Example:
        >>> dashboard = AttackMetricsDashboard('benchmark_results')
        >>> dashboard.create_dashboard('dashboard.html')
        >>> stats = dashboard.generate_summary_statistics()
    """
```

##### `BenchmarkResult`
```python
@dataclass
class BenchmarkResult:
    """
    Result from a single benchmark run.
    
    Attributes:
        attack_name (str): Name of the attack
        attack_type (str): Type of attack
        model_name (str): Name of the model
        verifier_name (str): Name of the verifier
        success (bool): Whether attack succeeded
        confidence (float): Attack confidence score
        execution_time (float): Time taken in seconds
        memory_usage (float): Memory used in MB
        far_before (float): FAR before attack
        far_after (float): FAR after attack
        frr_before (float): FRR before attack
        frr_after (float): FRR after attack
        accuracy_before (float): Accuracy before attack
        accuracy_after (float): Accuracy after attack
        defense_detected (bool): Whether defense detected attack
        defense_confidence (float): Defense confidence score
        metadata (Dict[str, Any]): Additional metadata
    """
```

### Utility Functions

#### `run_standard_benchmark`
```python
def run_standard_benchmark(model: nn.Module,
                          verifier: Any,
                          data_loader: DataLoader,
                          device: str = 'cpu',
                          save_results: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Run standard benchmark suite and generate report.
    
    Parameters:
        model: Model to benchmark
        verifier: Verifier to test
        data_loader: Data for evaluation
        device: Device to run on
        save_results: Whether to save results
        
    Returns:
        Tuple of (results DataFrame, report dictionary)
        
    Example:
        >>> results, report = run_standard_benchmark(
        ...     model, verifier, test_data
        ... )
        >>> print(f"Robustness: {report['summary']['robustness_score']:.1f}")
    """
```

#### `create_comparison_dashboard`
```python
def create_comparison_dashboard(results_dict: Dict[str, pd.DataFrame],
                               output_dir: str = "benchmark_reports") -> None:
    """
    Create comparison dashboard for multiple models.
    
    Parameters:
        results_dict: Dictionary mapping model names to results
        output_dir: Directory to save dashboard files
        
    Creates:
        - leaderboard.csv: Model ranking
        - all_results.csv: Combined results
        - dashboard.html: Interactive dashboard
        
    Example:
        >>> results = {
        ...     'Model_A': benchmark.run_benchmark(model_a, verifier, data),
        ...     'Model_B': benchmark.run_benchmark(model_b, verifier, data)
        ... }
        >>> create_comparison_dashboard(results)
    """
```

## Error Handling

### Custom Exceptions

```python
class AttackConfigError(Exception):
    """Raised when attack configuration is invalid."""
    pass

class DefenseConfigError(Exception):
    """Raised when defense configuration is invalid."""
    pass

class BenchmarkError(Exception):
    """Raised when benchmark execution fails."""
    pass

class VerificationError(Exception):
    """Raised when verification fails."""
    pass
```

### Error Handling Examples

```python
from pot.core.attack_suites import AttackConfig, AttackConfigError

try:
    config = AttackConfig(
        name="test",
        attack_type="invalid_type",  # Invalid
        budget={},
        strength="weak",
        success_metrics={}
    )
except AttackConfigError as e:
    print(f"Configuration error: {e}")
    # Use default configuration
    config = AttackConfig.default()
```

## Type Hints

### Common Types

```python
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Type aliases
AttackResult = Dict[str, Any]
DefenseResult = Dict[str, Any]
BenchmarkResults = pd.DataFrame
RobustnessScore = float
ThreatLevel = float
```

### Function Signatures

```python
def verify_model(
    model: nn.Module,
    verifier: Optional[BaseVerifier] = None,
    attacks: Optional[List[AttackConfig]] = None,
    defenses: Optional[DefenseConfig] = None,
    data: Optional[DataLoader] = None,
    device: Union[str, torch.device] = 'cpu',
    verbose: bool = True
) -> Tuple[bool, RobustnessScore, BenchmarkResults]:
    """
    Comprehensive model verification with attacks and defenses.
    
    Returns:
        Tuple of (verified, robustness_score, detailed_results)
    """
    pass
```

## Configuration Schema

### YAML Schema

```yaml
# attack_config.yaml schema
type: object
properties:
  attacks:
    type: object
    properties:
      distillation:
        type: object
        required: [temperatures, alpha_values, default_epochs]
      compression:
        type: object
        required: [pruning_ratios, quantization_bits]
      budgets:
        type: object
        required: [weak, moderate, strong]
  defense:
    type: object
    properties:
      adaptive:
        type: object
        required: [update_frequency, threshold_adjustment]
      filtering:
        type: object
        required: [detection_methods, sanitization]
      randomization:
        type: object
        required: [levels, methods]
required: [attacks, defense]
```

### Validation

```python
import jsonschema
import yaml

def validate_config(config_path: str, schema_path: str) -> bool:
    """Validate configuration against schema."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    
    try:
        jsonschema.validate(config, schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"Configuration invalid: {e}")
        return False
```

## Migration Guide

### From v1.0 to v2.0

```python
# v1.0 (deprecated)
from pot.attacks import DistillationAttack
attack = DistillationAttack(temperature=5.0)
result = attack.run(model, data)

# v2.0 (current)
from pot.core.attack_suites import AttackConfig, StandardAttackSuite
config = AttackConfig(
    name="distillation",
    attack_type="distillation",
    parameters={'temperature': 5.0},
    budget={'queries': 1000},
    strength='moderate',
    success_metrics={'accuracy_drop': 0.1}
)
suite = StandardAttackSuite()
result = suite.execute_attack(config, model, data)
```

## Performance Considerations

### Memory Management

```python
# Efficient batch processing
def process_large_dataset(model, data_loader, attack_config):
    results = []
    torch.cuda.empty_cache()  # Clear cache
    
    with torch.no_grad():  # Disable gradients when not needed
        for batch_idx, (data, labels) in enumerate(data_loader):
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()  # Periodic cleanup
            
            result = process_batch(model, data, attack_config)
            results.append(result)
    
    return results
```

### Optimization Tips

1. **Use GPU acceleration** when available
2. **Batch processing** for efficiency
3. **Cache intermediate results**
4. **Use mixed precision** for memory savings
5. **Profile code** to identify bottlenecks

```python
# Example optimization
import torch.cuda.amp as amp

# Mixed precision training
scaler = amp.GradScaler()
with amp.autocast():
    output = model(input_data)
    loss = criterion(output, target)
```

## Testing

### Unit Tests

```python
import unittest
from pot.core.attack_suites import AttackConfig

class TestAttackConfig(unittest.TestCase):
    def test_valid_config(self):
        config = AttackConfig(
            name="test",
            attack_type="distillation",
            budget={'queries': 1000},
            strength='moderate',
            success_metrics={'accuracy_drop': 0.1}
        )
        self.assertTrue(config.validate())
    
    def test_invalid_strength(self):
        with self.assertRaises(ValueError):
            AttackConfig(
                name="test",
                attack_type="distillation",
                budget={},
                strength='invalid',  # Invalid
                success_metrics={}
            )
```

### Integration Tests

```python
def test_end_to_end_pipeline():
    """Test complete attack-defense pipeline."""
    model = create_test_model()
    verifier = create_test_verifier()
    data = create_test_data()
    
    # Run attack
    attack_result = run_attack(model, data)
    assert attack_result['success'] in [True, False]
    
    # Apply defense
    defense_result = apply_defense(attack_result, model)
    assert 'verified' in defense_result
    
    # Benchmark
    score = compute_robustness(model, verifier, data)
    assert 0 <= score <= 100
```