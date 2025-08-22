# Attack Benchmarking and Metrics

Comprehensive benchmarking suite for evaluating model robustness against attacks and defense effectiveness.

## Overview

The benchmarking system provides:
- **Standardized attack suite** with 20+ predefined attacks
- **Automated benchmarking** with performance metrics
- **Model comparison** and leaderboard generation
- **Interactive dashboards** for visualization
- **Robustness scoring** (0-100 scale)

## Quick Start

```python
from pot.eval.attack_benchmarks import run_standard_benchmark

# Run standard benchmark
results, report = run_standard_benchmark(
    model=your_model,
    verifier=your_verifier,
    data_loader=your_data,
    device='cuda'
)

print(f"Robustness Score: {report['summary']['robustness_score']:.1f}/100")
```

## Components

### 1. AttackBenchmark

Main benchmarking class for systematic evaluation.

```python
from pot.eval.attack_benchmarks import AttackBenchmark

benchmark = AttackBenchmark(
    device='cuda',
    verbose=True,
    save_results=True,
    results_dir='benchmark_results'
)

# Run benchmark with standard attacks
results = benchmark.run_benchmark(
    model=model,
    verifier=verifier,
    data_loader=data_loader,
    attack_names=None,  # Use all standard attacks
    include_defenses=True
)

# Compute robustness score
score = benchmark.compute_robustness_score(results)
```

### 2. Standard Attack Suite

Predefined attacks for consistent evaluation:

| Attack Type | Variants | Description |
|------------|----------|-------------|
| Distillation | weak, moderate, strong | Knowledge distillation attacks |
| Compression | 30%, 50%, 70% pruning | Model compression attacks |
| Quantization | 8-bit, 4-bit, 2-bit | Precision reduction attacks |
| Wrapper | naive, adaptive | Model wrapping attacks |
| Fine-tuning | minimal, targeted | Transfer learning attacks |
| Adversarial | small/large patches | Visual adversarial attacks |
| Extraction | Jacobian-based | Model extraction attacks |
| Backdoor | simple triggers | Backdoor injection |

### 3. AttackMetricsDashboard

Interactive visualization of benchmark results.

```python
from pot.eval.attack_benchmarks import AttackMetricsDashboard

dashboard = AttackMetricsDashboard('benchmark_results')

# Create full dashboard
dashboard.create_dashboard('dashboard.html')

# Individual visualizations
dashboard.plot_attack_success_rates()
dashboard.plot_far_frr_tradeoffs()
dashboard.plot_defense_adaptation()
```

## Benchmark Metrics

### Core Metrics

1. **Robustness Score** (0-100)
   - Weighted combination of:
     - Attack resistance (30%)
     - Performance retention (25%)
     - Defense effectiveness (25%)
     - Efficiency (20%)

2. **Performance Metrics**
   - False Acceptance Rate (FAR)
   - False Rejection Rate (FRR)
   - Accuracy retention
   - Execution time
   - Memory usage

3. **Defense Metrics**
   - Detection rate
   - Defense confidence
   - Adaptation effectiveness

### BenchmarkResult Structure

```python
@dataclass
class BenchmarkResult:
    attack_name: str
    attack_type: str
    model_name: str
    verifier_name: str
    success: bool
    confidence: float
    execution_time: float
    memory_usage: float
    far_before: float
    far_after: float
    frr_before: float
    frr_after: float
    accuracy_before: float
    accuracy_after: float
    defense_detected: bool
    defense_confidence: float
    metadata: Dict[str, Any]
```

## Model Comparison

### Leaderboard Generation

```python
# Compare multiple models
results_dict = {
    'Model_A': benchmark.run_benchmark(model_a, verifier, data),
    'Model_B': benchmark.run_benchmark(model_b, verifier, data),
    'Model_C': benchmark.run_benchmark(model_c, verifier, data)
}

# Generate leaderboard
leaderboard = benchmark.generate_leaderboard(
    results_dict,
    save_path='leaderboard.csv'
)
```

### Leaderboard Format

| Rank | Model | Robustness Score | Success Rate | FAR Increase | FRR Increase | Detection Rate |
|------|-------|-----------------|--------------|--------------|--------------|----------------|
| 1 | Model_C | 85.2 | 12.3% | 0.021 | 0.015 | 87.5% |
| 2 | Model_A | 72.1 | 28.7% | 0.045 | 0.032 | 71.2% |
| 3 | Model_B | 61.5 | 41.2% | 0.078 | 0.056 | 58.3% |

## Visualization

### Dashboard Features

1. **Attack Success Rates**
   - Bar chart by attack type
   - Error bars showing variance
   - Sample size indicators

2. **FAR/FRR Trade-offs**
   - ROC curves before/after attacks
   - Attack-specific impacts
   - Baseline comparison

3. **Defense Adaptation**
   - Time-series detection rates
   - Rolling average trends
   - Attack type breakdown

4. **Performance Impact**
   - Accuracy degradation
   - Latency increases
   - Memory overhead

### Creating Custom Visualizations

```python
import plotly.graph_objects as go

# Custom visualization from results
fig = go.Figure()

# Add traces for each attack type
for attack_type in results['attack_type'].unique():
    data = results[results['attack_type'] == attack_type]
    fig.add_trace(go.Box(
        y=data['confidence'],
        name=attack_type
    ))

fig.update_layout(
    title='Attack Confidence Distribution',
    yaxis_title='Confidence Score'
)
fig.show()
```

## Running Benchmarks

### Quick Benchmark

```python
# Minimal benchmark with subset of attacks
quick_attacks = [
    "distillation_weak",
    "pruning_50",
    "quantization_4bit"
]

results = benchmark.run_benchmark(
    model=model,
    verifier=verifier,
    data_loader=data_loader,
    attack_names=quick_attacks
)
```

### Comprehensive Benchmark

```python
# Full benchmark suite
results = benchmark.run_benchmark(
    model=model,
    verifier=verifier,
    data_loader=data_loader,
    attack_names=AttackBenchmark.STANDARD_ATTACKS,
    include_defenses=True
)

# Generate detailed report
report = benchmark.generate_report(
    results,
    save_path='comprehensive_report.json'
)
```

### Custom Attack Configuration

```python
from pot.core.attack_suites import AttackConfig

# Define custom attack
custom_attack = AttackConfig(
    name="custom_distillation",
    attack_type="distillation",
    budget={'queries': 5000, 'compute_time': 300},
    strength='adaptive',
    success_metrics={'accuracy_drop': 0.15},
    parameters={'temperature': 7.0, 'epochs': 100}
)

# Add to benchmark
benchmark.STANDARD_ATTACKS.append("custom_distillation")
```

## Report Generation

### JSON Report Structure

```json
{
  "summary": {
    "total_attacks": 20,
    "successful_attacks": 5,
    "success_rate": 0.25,
    "robustness_score": 75.3,
    "avg_execution_time": 12.5,
    "total_execution_time": 250.0
  },
  "by_attack_type": {
    "distillation": {
      "count": 3,
      "success_rate": 0.33,
      "avg_confidence": 0.65
    }
  },
  "performance_impact": {
    "avg_far_increase": 0.032,
    "avg_frr_increase": 0.021,
    "avg_accuracy_drop": 0.08
  },
  "defense_effectiveness": {
    "detection_rate": 0.82,
    "avg_confidence": 0.71
  },
  "recommendations": [
    "Model shows good robustness against tested attacks",
    "Vulnerable to compression attacks (success rate: 75%)"
  ]
}
```

## Best Practices

### 1. Data Preparation
- Use representative test data
- Ensure sufficient sample size (100+ samples)
- Balance classes for fair evaluation

### 2. Attack Selection
- Start with quick subset for development
- Run full suite for production evaluation
- Include attack variants (weak/strong)

### 3. Performance Optimization
- Use GPU when available
- Batch processing for efficiency
- Cache results for comparison

### 4. Interpretation
- **Score > 80**: Excellent robustness
- **Score 60-80**: Good robustness
- **Score 40-60**: Moderate robustness
- **Score < 40**: Poor robustness

## Command Line Usage

```bash
# Run example benchmarking
python example_benchmarking.py

# Quick evaluation
python -c "from pot.eval.attack_benchmarks import run_standard_benchmark; run_standard_benchmark(model, verifier, data)"
```

## Integration with CI/CD

```python
# Automated benchmark in CI pipeline
def test_model_robustness():
    results, report = run_standard_benchmark(
        model, verifier, test_data
    )
    
    # Assert minimum robustness
    assert report['summary']['robustness_score'] >= 70.0
    assert report['summary']['success_rate'] <= 0.30
    
    # Check specific vulnerabilities
    for attack_type, metrics in report['by_attack_type'].items():
        assert metrics['success_rate'] <= 0.50, f"Vulnerable to {attack_type}"
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use subset of attacks
   - Process sequentially

2. **Slow Execution**
   - Use GPU acceleration
   - Reduce attack iterations
   - Cache intermediate results

3. **Missing Visualizations**
   - Install plotly: `pip install plotly`
   - Check results directory exists
   - Verify data format

## Future Enhancements

- [ ] Distributed benchmarking
- [ ] Real-time monitoring
- [ ] Attack chain evaluation
- [ ] Automated hyperparameter tuning
- [ ] Cross-model transfer testing
- [ ] Certification generation

## References

- [Robustness Metrics](https://arxiv.org/abs/robustness)
- [Attack Taxonomies](https://arxiv.org/abs/attacks)
- [Defense Benchmarks](https://arxiv.org/abs/defenses)