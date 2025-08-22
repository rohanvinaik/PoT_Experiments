# Vision Verifier Module

## Overview

The Vision Verifier provides comprehensive verification for vision models using frequency-domain challenges, texture patterns, and natural image synthesis. It implements the Proof-of-Training verification protocol specifically designed for computer vision models, enabling detection of model substitution, wrapper attacks, and behavioral deviations.

## Features

### Challenge Types

#### Frequency Challenges
- **Low Frequency**: Gradients, large-scale patterns for testing model sensitivity to global structures
- **Mid Frequency**: Edges, regular patterns for evaluating edge detection capabilities  
- **High Frequency**: Fine details, noise patterns for assessing texture discrimination
- **Mixed Frequency**: Combinations of different bands for comprehensive frequency response testing

#### Texture Challenges
- **Perlin Noise**: Natural-looking random textures with controllable octaves and persistence
- **Voronoi Diagrams**: Cell-based patterns with customizable point density and coloring
- **Fractals**: Julia and Mandelbrot sets for testing geometric pattern recognition
- **Gabor Filters**: Orientation-selective patterns for spatial frequency analysis

#### Natural Challenges
- **Synthetic Landscapes**: Procedural terrain generation with varying complexity
- **Cloud Patterns**: Procedural cloud synthesis using noise functions
- **Abstract Compositions**: Geometric abstractions with controlled randomness

### Verification Methods

#### Sequential Testing (SPRT)
- Early stopping based on statistical confidence intervals
- Adaptive challenge selection with empirical Bernstein bounds
- Efficient for quick verification with 90% average speedup
- Configurable Type I and Type II error rates

#### Batch Testing
- Fixed number of challenges for comprehensive evaluation
- Better for detailed analysis and calibration
- Supports parallel processing for improved throughput
- Comprehensive statistical validation

### Distance Metrics

#### Logit-based Distances
- **KL Divergence**: Measures distributional differences in model outputs
- **JS Divergence**: Symmetric version of KL divergence for stable comparisons
- **Wasserstein Distance**: Earth mover's distance for probability distributions
- **L2 Distance**: Euclidean distance in logit space

#### Embedding-based Distances
- **Cosine Similarity**: Angular similarity in high-dimensional embeddings
- **Euclidean Distance**: L2 distance in embedding space
- **Centered Kernel Alignment (CKA)**: Measures representation similarity
- **Maximum Mean Discrepancy (MMD)**: Distribution comparison in RKHS

## Installation

```bash
# Install the PoT framework
pip install -r requirements.txt

# Verify installation
python -c "from pot.vision.verifier import VisionVerifier; print('✓ Vision verifier installed')"
```

## Quick Start

### Basic Verification

```python
from pot.vision.verifier import EnhancedVisionVerifier
import torchvision.models as models

# Load model
model = models.resnet18(pretrained=True)

# Create verifier with default configuration
config = {
    'device': 'cuda',
    'verification_method': 'batch',
    'temperature': 1.0
}
verifier = EnhancedVisionVerifier(model, config)

# Run verification
result = verifier.verify_session(
    num_challenges=10,
    challenge_types=['frequency', 'texture']
)

print(f"Verified: {result['verified']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Success Rate: {result['success_rate']:.2%}")
```

### Custom Configuration

```python
from pot.vision.vision_config import VisionVerifierConfig

# Create custom configuration
config = VisionVerifierConfig(
    num_challenges=20,
    challenge_types=['frequency', 'texture', 'natural'],
    verification_method='sequential',
    distance_metric='combined',
    normalization='softmax',
    temperature=1.5,
    image_size=(224, 224),
    device='cuda'
)

# Use configuration with verifier
verifier = EnhancedVisionVerifier(model, config.to_dict())
```

### Calibration

```python
from pot.vision.verifier import VisionVerifierCalibrator

# Create calibrator
calibrator = VisionVerifierCalibrator(verifier)

# Calibrate on genuine model
stats = calibrator.calibrate(
    num_samples=100,
    challenge_types=['frequency', 'texture']
)

# Save calibration for future use
calibrator.save_calibration('model_calibration.json')

# Validate calibration quality
validation_results = calibrator.validate_calibration(
    num_validation_samples=50
)
print(f"Validation success rate: {validation_results['overall']:.2%}")
```

### Benchmarking

```python
from pot.vision.benchmark import VisionBenchmark

# Create benchmark suite
benchmark = VisionBenchmark(device='cuda')

# Run comprehensive benchmark
results = benchmark.run_benchmark(
    model=model,
    model_name='ResNet18',
    benchmark_level='comprehensive',
    calibrate=True,
    measure_memory=True
)

# Generate detailed report
report_path = benchmark.generate_report(
    results=results,
    output_path='resnet18_benchmark.html',
    include_plots=True
)
print(f"Report saved to: {report_path}")
```

### Model Comparison

```python
# Compare multiple models
models = {
    'ResNet18': models.resnet18(pretrained=True),
    'ResNet50': models.resnet50(pretrained=True),
    'EfficientNet': models.efficientnet_b0(pretrained=True)
}

# Run comparison benchmark
comparison_results = benchmark.compare_models(
    models=models,
    benchmark_level='intermediate',
    calibrate=True
)

# View performance ranking
model_summary = comparison_results.groupby('model_name').agg({
    'success_rate': 'mean',
    'confidence': 'mean',
    'throughput': 'mean'
}).sort_values('success_rate', ascending=False)

print("Model Performance Ranking:")
print(model_summary)
```

### Robustness Evaluation

```python
from pot.vision.benchmark import VisionRobustnessEvaluator

# Create robustness evaluator
evaluator = VisionRobustnessEvaluator(verifier)

# Test noise robustness
noise_results = evaluator.evaluate_noise_robustness(
    noise_levels=[0.01, 0.05, 0.1, 0.2],
    num_trials=20,
    challenge_types=['frequency', 'texture']
)

# Test transformation robustness
transform_results = evaluator.evaluate_transformation_robustness(
    num_trials=20,
    challenge_types=['frequency', 'texture']
)

# Test adversarial robustness
adv_results = evaluator.evaluate_adversarial_robustness(
    epsilon_values=[0.01, 0.03, 0.1],
    attack_steps=10,
    num_trials=10
)

# Generate robustness report
evaluator.generate_robustness_report(
    results={**noise_results, **transform_results, **adv_results},
    output_path='robustness_report.html'
)
```

## Architecture Support

The verifier automatically detects and adapts to common architectures:

- **ResNet family** (ResNet18, ResNet50, ResNet101, etc.)
- **Vision Transformers (ViT)** with automatic patch extraction
- **EfficientNet** with compound scaling detection
- **DenseNet** with dense connectivity handling
- **Generic CNNs** with automatic probe point detection

### Custom Architecture Support

```python
# For custom architectures, specify probe points manually
custom_config = {
    'probe_points': {
        'early': 'features.0',
        'mid': 'features.8', 
        'late': 'features.16',
        'final': 'classifier'
    }
}

verifier = EnhancedVisionVerifier(custom_model, custom_config)
```

## CLI Usage

The vision verifier includes a comprehensive command-line interface:

```bash
# Basic verification
python -m pot.vision.cli verify --model resnet18 --num-challenges 10

# Custom challenges
python -m pot.vision.cli verify \
    --model path/to/model.pt \
    --challenge-types frequency texture \
    --method sequential \
    --output results.json

# List available presets
python -m pot.vision.cli list-presets

# Create configuration template
python -m pot.vision.cli create-config \
    --preset comprehensive \
    --output my_config.yaml

# Validate configuration
python -m pot.vision.cli validate-config my_config.yaml

# Model information
python -m pot.vision.cli model-info --model resnet18

# Show results
python -m pot.vision.cli show-results results.json
```

## Configuration Reference

### VisionVerifierConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_challenges` | int | 10 | Number of challenges to generate |
| `challenge_types` | List[str] | ['frequency', 'texture'] | Types of challenges to use |
| `verification_method` | str | 'batch' | 'batch' or 'sequential' |
| `distance_metric` | str | 'combined' | Distance metric for comparison |
| `normalization` | str | 'softmax' | Logit normalization method |
| `temperature` | float | 1.0 | Temperature scaling parameter |
| `image_size` | Tuple[int, int] | (224, 224) | Input image dimensions |
| `device` | str | 'cuda' | Device for computation |
| `confidence_threshold` | float | 0.95 | Confidence threshold for verification |
| `early_stopping` | bool | True | Enable early stopping in sequential mode |

### Preset Configurations

```python
from pot.vision.vision_config import VisionConfigPresets

# Quick verification (fast, basic confidence)
quick_config = VisionConfigPresets.quick_verification()

# Standard verification (balanced speed/accuracy)
standard_config = VisionConfigPresets.standard_verification()

# Comprehensive verification (thorough, high confidence)
comprehensive_config = VisionConfigPresets.comprehensive_verification()

# Research configuration (maximum detail)
research_config = VisionConfigPresets.research_verification()

# Production configuration (optimized for deployment)
production_config = VisionConfigPresets.production_verification()
```

## Performance Considerations

### Optimization Tips

1. **GPU Acceleration**: Use CUDA for faster inference
   ```python
   config = {'device': 'cuda'}
   ```

2. **Batch Processing**: Enable batch mode for throughput optimization
   ```python
   config = {'verification_method': 'batch', 'batch_size': 32}
   ```

3. **Challenge Pre-generation**: Cache challenges for reduced latency
   ```python
   from pot.vision.datasets import create_verification_dataloader
   
   dataloader = create_verification_dataloader(
       num_samples=1000,
       cache_challenges=True
   )
   ```

4. **Reference Statistics Caching**: Load pre-computed calibration
   ```python
   calibrator.load_calibration('cached_stats.json')
   ```

5. **Image Resolution**: Adjust for speed/accuracy tradeoff
   ```python
   config = {'image_size': (128, 128)}  # Faster but less detailed
   ```

### Performance Benchmarks

| Configuration | Speed | Accuracy | Memory | Use Case |
|---------------|-------|----------|--------|----------|
| Quick | ~1s | 85% | Low | Development/Testing |
| Standard | ~5s | 90% | Medium | General Purpose |
| Comprehensive | ~30s | 95% | High | Production/Research |

## Troubleshooting

### Common Issues

#### Out of Memory Error
```python
# Reduce batch size
config = {'batch_size': 8}

# Use smaller images
config = {'image_size': (128, 128)}

# Use CPU if necessary
config = {'device': 'cpu'}
```

#### Slow Verification
```python
# Use sequential testing with early stopping
config = {
    'verification_method': 'sequential',
    'early_stopping': True,
    'confidence_threshold': 0.9
}
```

#### Poor Calibration Results
```python
# Increase calibration samples
calibrator.calibrate(num_samples=500)

# Use multiple challenge types
calibrator.calibrate(challenge_types=['frequency', 'texture', 'natural'])
```

#### Architecture Not Detected
```python
# Manually specify probe points
config = {
    'probe_points': {
        'layer1': 'features.0',
        'layer2': 'features.4',
        'layer3': 'features.8'
    }
}
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run verification with debug info
result = verifier.verify_session(
    num_challenges=5,
    challenge_types=['frequency']
)
```

### Validation Checks

Verify your setup with built-in checks:

```python
# Check model compatibility
from pot.vision.utils import check_model_compatibility
compatibility = check_model_compatibility(model)
print(f"Compatible: {compatibility['compatible']}")

# Validate configuration
from pot.vision.vision_config import validate_config
errors = validate_config(config)
if errors:
    print("Configuration errors:", errors)
```

## API Reference

### Core Classes

- **`EnhancedVisionVerifier`**: Main verification class with batch processing
- **`VisionVerifierCalibrator`**: Calibration and reference statistics management
- **`VisionBenchmark`**: Standardized benchmarking suite
- **`VisionRobustnessEvaluator`**: Robustness testing against perturbations

### Utility Functions

- **`create_verification_dataloader()`**: Create challenge datasets
- **`preprocess_image()`**: Image preprocessing utilities
- **`compute_activation_statistics()`**: Activation analysis tools

### Configuration Classes

- **`VisionVerifierConfig`**: Main configuration dataclass
- **`VisionConfigPresets`**: Pre-defined configuration templates

## Examples

Complete examples are available in the `examples/` directory:

- **`vision_verification_demo.ipynb`**: Jupyter notebook with interactive demo
- **`basic_verification.py`**: Simple verification script
- **`model_comparison.py`**: Multi-model benchmarking example
- **`robustness_analysis.py`**: Comprehensive robustness evaluation
- **`custom_challenges.py`**: Creating custom challenge types

## Citation

If you use the Vision Verifier in your research, please cite:

```bibtex
@article{pot_vision_2024,
  title={Proof-of-Training for Vision Models: Cryptographic Verification of Neural Network Training},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](../LICENSE) for details.

## Support

- **Documentation**: [Full API docs](https://pot-framework.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/pot-framework/pot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pot-framework/pot/discussions)
- **Email**: pot-support@example.com

## Changelog

### Version 1.0.0
- Initial release with comprehensive vision verification
- Support for frequency, texture, and natural challenges
- Sequential and batch verification methods
- Automated calibration and benchmarking
- Robustness evaluation suite

### Version 1.1.0 (Planned)
- Additional challenge types (adversarial, style transfer)
- Real-time verification monitoring
- Distributed verification across multiple GPUs
- Integration with popular model registries