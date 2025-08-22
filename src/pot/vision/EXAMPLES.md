# Vision Verification Examples

This directory contains comprehensive examples demonstrating the capabilities of the PoT Vision Verification framework.

## Example Files

### 📓 [vision_verification_demo.ipynb](examples/vision_verification_demo.ipynb)
**Jupyter Notebook with Interactive Demo**

Complete interactive demonstration covering:
- Model setup and configuration
- Challenge generation and visualization
- Verification execution and analysis
- Calibration procedures
- Benchmarking and comparison
- Robustness evaluation
- Report generation

**Usage:**
```bash
jupyter notebook pot/vision/examples/vision_verification_demo.ipynb
```

### 🔰 [basic_verification.py](examples/basic_verification.py) 
**Simple Verification Example**

The simplest way to verify a vision model with minimal setup.

**Features:**
- Model creation and configuration
- Basic verification execution
- Result interpretation
- Next steps guidance

**Usage:**
```bash
python pot/vision/examples/basic_verification.py
```

**Expected Output:**
```
Basic Vision Verification Example
==================================================

1. Creating model...
   Model: Sequential
   Parameters: 5,418

2. Configuring verifier...
   Configuration: {'device': 'cpu', 'verification_method': 'batch'}

3. Creating verifier...
   Verifier created successfully

4. Running verification...
   Status: PASSED/FAILED
   Confidence: XX.X%
   Success Rate: XX.X%
```

### 🏆 [model_comparison.py](examples/model_comparison.py)
**Multi-Model Benchmarking**

Comprehensive comparison of multiple vision models using standardized benchmarks.

**Features:**
- Multiple model architecture creation
- Performance benchmarking
- Statistical analysis
- Performance ranking
- Resource efficiency analysis
- Automated report generation

**Usage:**
```bash
python pot/vision/examples/model_comparison.py
```

**Output Files:**
- `detailed_results.csv` - Complete benchmark data
- `model_ranking.csv` - Performance rankings
- `model_info.csv` - Architecture information

### 🛡️ [robustness_analysis.py](examples/robustness_analysis.py)
**Comprehensive Robustness Evaluation**

In-depth robustness testing against various perturbations and attacks.

**Features:**
- Noise robustness evaluation
- Transformation robustness testing
- Adversarial attack simulation
- Vulnerability identification
- Statistical analysis
- Visualization generation

**Usage:**
```bash
python pot/vision/examples/robustness_analysis.py
```

**Output Files:**
- `robustness_report.json` - Comprehensive analysis
- `robustness_analysis.png` - Visual summary
- `noise_robustness.csv` - Noise test results
- `transformation_robustness.csv` - Transform test results

### 🎨 [custom_challenges.py](examples/custom_challenges.py)
**Custom Challenge Creation**

Demonstrates how to create and integrate custom challenge types.

**Features:**
- Edge detection challenges
- Color constancy testing
- Motion blur simulation
- Depth of field effects
- Integration examples
- Visualization tools

**Usage:**
```bash
python pot/vision/examples/custom_challenges.py
```

**Custom Challenge Types:**
- **Edge Detection**: Sobel filter-based edge enhancement
- **Color Constancy**: Different illumination conditions
- **Motion Blur**: Directional blur effects
- **Depth of Field**: Focus and blur simulation

## Quick Start Guide

### 1. Basic Verification
```python
from pot.vision.verifier import EnhancedVisionVerifier

# Create verifier
config = {'device': 'cpu', 'verification_method': 'batch'}
verifier = EnhancedVisionVerifier(model, config)

# Run verification
result = verifier.verify_session(
    num_challenges=10,
    challenge_types=['frequency', 'texture']
)

print(f"Verified: {result['verified']}")
```

### 2. Model Benchmarking
```python
from pot.vision.benchmark import VisionBenchmark

# Create benchmark suite
benchmark = VisionBenchmark()

# Compare models
results = benchmark.compare_models({
    'ModelA': model_a,
    'ModelB': model_b
}, benchmark_level='intermediate')

# Generate report
benchmark.generate_report(results, 'comparison_report.html')
```

### 3. Robustness Testing
```python
from pot.vision.benchmark import VisionRobustnessEvaluator

# Create evaluator
evaluator = VisionRobustnessEvaluator(verifier)

# Test robustness
noise_results = evaluator.evaluate_noise_robustness()
transform_results = evaluator.evaluate_transformation_robustness()

# Generate report
evaluator.generate_robustness_report(
    results={**noise_results, **transform_results},
    output_path='robustness_report.html'
)
```

### 4. Custom Challenges
```python
class MyChallenger:
    def generate_my_challenge(self, size=(224, 224)):
        # Custom challenge logic
        return challenge_tensor

# Integrate with verifier
challenger = MyChallenger()
challenge = challenger.generate_my_challenge()
output = verifier.run_model(challenge.unsqueeze(0))
```

## Configuration Examples

### Quick Verification
```python
config = {
    'device': 'cpu',
    'verification_method': 'sequential',
    'num_challenges': 5,
    'challenge_types': ['frequency']
}
```

### Comprehensive Analysis
```python
config = {
    'device': 'cuda',
    'verification_method': 'batch',
    'num_challenges': 50,
    'challenge_types': ['frequency', 'texture', 'natural'],
    'temperature': 1.0,
    'normalization': 'softmax'
}
```

### Production Deployment
```python
config = {
    'device': 'cuda',
    'verification_method': 'sequential',
    'confidence_threshold': 0.95,
    'early_stopping': True,
    'max_challenges': 20
}
```

## Expected Outputs

### Verification Results
```json
{
  "verified": true,
  "confidence": 0.92,
  "success_rate": 0.88,
  "num_challenges": 10,
  "challenge_types": ["frequency", "texture"],
  "total_time": 2.45
}
```

### Benchmark Results
```
Model Performance Ranking:
  1. ResNetLike: 0.856 (composite score)
     Success Rate: 85.6%
     Confidence: 89.2%
     Throughput: 45.2 challenges/sec
  
  2. Standard: 0.742 (composite score)
     Success Rate: 78.1%
     Confidence: 81.4%
     Throughput: 52.1 challenges/sec
```

### Robustness Analysis
```
Robustness Summary:
  Average robustness score: 0.673
  Most vulnerable to:
    - gaussian_blur (robustness: 0.234)
    - contrast_increase (robustness: 0.312)
  Most robust against:
    - flip_horizontal (robustness: 0.891)
    - rotation_90 (robustness: 0.856)
```

## Common Issues and Solutions

### Memory Issues
```python
# Reduce batch size and image resolution
config = {
    'batch_size': 8,
    'image_size': (128, 128),
    'device': 'cpu'
}
```

### Performance Optimization
```python
# Use sequential testing with early stopping
config = {
    'verification_method': 'sequential',
    'early_stopping': True,
    'confidence_threshold': 0.9
}
```

### Calibration Improvements
```python
# Increase calibration samples
calibrator.calibrate(
    num_samples=500,
    challenge_types=['frequency', 'texture', 'natural']
)
```

## Best Practices

### 1. Development Workflow
1. Start with `basic_verification.py` to test setup
2. Use `model_comparison.py` to evaluate multiple models
3. Run `robustness_analysis.py` for security assessment
4. Create custom challenges for specialized testing

### 2. Production Deployment
1. Use calibrated verifiers with sufficient samples (500+)
2. Enable GPU acceleration for performance
3. Implement monitoring and alerting
4. Regular robustness testing

### 3. Research Applications
1. Explore custom challenge types
2. Analyze failure modes
3. Compare different architectures
4. Study robustness characteristics

## Performance Benchmarks

| Example | Runtime | Memory | Use Case |
|---------|---------|--------|----------|
| Basic Verification | ~5s | <1GB | Quick testing |
| Model Comparison | ~30s | <2GB | Architecture analysis |
| Robustness Analysis | ~120s | <3GB | Security assessment |
| Custom Challenges | ~10s | <1GB | Specialized testing |

## Extending the Examples

### Adding New Challenge Types
1. Create challenger class in `custom_challenges.py`
2. Implement generation methods
3. Add visualization support
4. Integrate with verifier

### Custom Metrics
1. Extend distance metric calculations
2. Add evaluation criteria
3. Update reporting functions
4. Validate with known models

### Integration with Existing Workflows
1. Adapt configuration formats
2. Add CLI support
3. Implement batch processing
4. Create monitoring hooks

## Support and Documentation

- **Main Documentation**: [pot/vision/README.md](README.md)
- **API Reference**: See docstrings in source files
- **Configuration Guide**: [pot/vision/vision_config.py](vision_config.py)
- **CLI Usage**: `python -m pot.vision.cli --help`

## Contributing

To contribute new examples:

1. Follow the existing code structure
2. Include comprehensive documentation
3. Add error handling and validation
4. Provide expected outputs
5. Test on multiple configurations

## License

These examples are part of the PoT framework and are subject to the same license terms.