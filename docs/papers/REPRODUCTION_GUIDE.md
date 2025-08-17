# Proof-of-Training Reproduction Guide

This guide provides detailed instructions for reproducing the experimental results from the Proof-of-Training paper. Follow these steps to verify the paper claims and understand the framework's capabilities.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Step-by-Step Reproduction](#step-by-step-reproduction)
4. [Expected Outputs](#expected-outputs)
5. [Interpreting Results](#interpreting-results)
6. [Common Issues and Solutions](#common-issues-and-solutions)
7. [Known Limitations](#known-limitations)
8. [Advanced Reproduction Scenarios](#advanced-reproduction-scenarios)

## Prerequisites

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores, NVIDIA GPU with 16GB+ VRAM
- **Optimal**: 32GB RAM, 16 CPU cores, NVIDIA A100 40GB GPU

### Software Requirements

- Python 3.10+
- CUDA 12.0+ (for GPU acceleration)
- Git
- Make (optional, for convenience commands)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PoT_Experiments.git
cd PoT_Experiments
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install fuzzy hashing support
pip install ".[fuzzy]"
```

### 4. Verify Installation

```bash
# Check core modules
python -c "from pot.experiments import ReportGenerator, ResultValidator; print('✓ Core modules installed')"

# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Run quick test
python -m pytest tests/test_reproducibility.py::TestReproducibilityPipeline::test_minimal_reproduction -xvs
```

### 5. Set Environment Variables for Determinism

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Add to your shell profile for persistence
echo 'export PYTHONHASHSEED=0' >> ~/.bashrc
echo 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' >> ~/.bashrc
```

## Step-by-Step Reproduction

### Stage 1: Minimal Reproduction (5-10 minutes)

This stage verifies the basic framework functionality with minimal computational requirements.

```bash
# 1. Run minimal reproduction
python scripts/run_reproduction.py --config configs/minimal_reproduce.yaml --output-dir outputs/minimal_reproduction

# 2. Generate report
python pot/experiments/report_generator.py \
    --results outputs/minimal_reproduction/results.json \
    --paper-claims configs/paper_claims.json \
    --output-dir outputs/minimal_reproduction/reports

# 3. Validate results
python pot/experiments/result_validator.py \
    --results outputs/minimal_reproduction/results.json \
    --claimed-metrics configs/paper_claims.json \
    --strict-mode false
```

Expected output:
```
✓ Verification completed: 10/10 models verified
✓ Metrics calculated: FAR=0.009, FRR=0.011, Accuracy=0.990
✓ Report generated: outputs/minimal_reproduction/reports/report.md
✓ Validation passed: Results within 10% tolerance of paper claims
```

### Stage 2: Standard Reproduction (30-60 minutes)

This stage runs the standard experimental protocol with moderate computational requirements.

```bash
# 1. Generate reference models
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml

# 2. Run verification experiments
python scripts/run_verify.py \
    --config configs/vision_cifar10.yaml \
    --challenge_family vision:texture \
    --n 256 \
    --seq eb \
    --store-trace

# 3. Generate plots
python scripts/run_plots.py \
    --exp_dir outputs/vision_cifar10 \
    --plot_type all

# 4. Generate comprehensive report
python pot/experiments/report_generator.py \
    --results outputs/vision_cifar10/results.json \
    --paper-claims configs/paper_claims.json \
    --generate-all
```

### Stage 3: Comprehensive Reproduction (2-4 hours)

This stage runs all experiments from the paper including attack scenarios and ablations.

```bash
# Run complete reproduction pipeline
bash run_all.sh

# Or run selectively:

# E1: Separation vs Query Budget
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1

# E2: Leakage Ablation
python scripts/run_attack.py \
    --config configs/lm_small.yaml \
    --attack targeted_finetune \
    --rho 0.25

# E3: Sequential vs Fixed
python scripts/run_verify_enhanced.py \
    --config configs/vision_cifar10.yaml \
    --alpha 0.01 \
    --beta 0.01 \
    --tau-id 0.01

# E4: Language Model Verification
python scripts/run_verify.py \
    --config configs/lm_small.yaml \
    --challenge_family lm:templates \
    --n 512

# E5: Behavioral Fingerprinting
python examples/fingerprinting_demo.py

# E6: Baseline Comparisons
python scripts/run_baselines.py --config configs/vision_cifar10.yaml
```

## Expected Outputs

### Directory Structure

After successful reproduction, you should see:

```
outputs/
├── minimal_reproduction/
│   ├── results.json              # Raw experimental data
│   ├── metrics.json              # Calculated metrics
│   └── reports/
│       ├── report.md            # Markdown report
│       ├── report.html          # Interactive HTML report
│       └── report_data.json    # Structured data
│
├── vision_cifar10/
│   ├── E1/                      # Experiment 1 results
│   │   ├── grid_results.jsonl
│   │   ├── roc.png
│   │   └── calibration.png
│   ├── E3/                      # Experiment 3 results
│   │   ├── verify.jsonl
│   │   └── sequential_trace.jsonl
│   └── references/              # Reference models
│       └── resnet18.ckpt
│
└── consolidated_results.json    # All results combined
```

### Key Metrics to Verify

| Metric | Expected Value | Tolerance | Verification Command |
|--------|---------------|-----------|---------------------|
| FAR | 0.01 | ±10% | `jq '.metrics.far' outputs/*/metrics.json` |
| FRR | 0.01 | ±10% | `jq '.metrics.frr' outputs/*/metrics.json` |
| Accuracy | 0.99 | ±1% | `jq '.metrics.accuracy' outputs/*/metrics.json` |
| AUROC | 0.99 | ±1% | `jq '.metrics.auroc' outputs/*/metrics.json` |
| Avg Queries | 10 | ±20% | `jq '.metrics.avg_queries' outputs/*/metrics.json` |
| Query Reduction | 70-90% | - | Compare sequential vs fixed results |

## Interpreting Results

### Understanding the Report

The generated report contains:

1. **Executive Summary**
   - High-level comparison with paper claims
   - Pass/fail status for each metric
   - Overall reproduction success rate

2. **Detailed Metrics**
   - FAR/FRR with confidence intervals
   - Query efficiency statistics
   - Performance benchmarks

3. **Discrepancy Analysis**
   - Automatic detection of significant deviations
   - Severity classification (minor/moderate/major)
   - Statistical significance tests

4. **Reconciliation Notes**
   - Explanations for observed differences
   - Suggested parameter adjustments
   - Environmental factors

### Visualization Guide

| Plot | What it Shows | How to Interpret |
|------|--------------|------------------|
| ROC Curve | FAR vs FRR trade-off | Higher AUC = better separation |
| Query Distribution | Stopping time histogram | Left-skewed = efficient early stopping |
| Confidence Intervals | Metric uncertainty | Narrower = more reliable |
| Calibration Plot | Threshold selection | Intersection = optimal threshold |
| Sequential Trajectory | Decision evolution | Faster convergence = better |

### Statistical Validation

The framework performs multiple statistical tests:

```python
# Check statistical significance
from pot.experiments import MetricsCalculator

calculator = MetricsCalculator()
metrics = calculator.calculate_with_confidence_intervals(
    predictions, labels, 
    n_bootstrap=1000,
    confidence=0.95
)

print(f"FAR: {metrics['far']:.3f} [{metrics['far_ci'][0]:.3f}, {metrics['far_ci'][1]:.3f}]")
```

## Common Issues and Solutions

### Issue 1: Results Don't Match Paper Claims

**Symptoms**: Metrics deviate >10% from expected values

**Solutions**:
1. Verify seed configuration:
   ```bash
   grep -r "seed" configs/minimal_reproduce.yaml
   # Should show: seed: 42
   ```

2. Check deterministic mode:
   ```python
   import torch
   torch.use_deterministic_algorithms(True)
   ```

3. Review reconciliation suggestions:
   ```bash
   cat outputs/*/reports/validation.json | jq '.reconciliation_suggestions'
   ```

### Issue 2: Out of Memory Errors

**Symptoms**: CUDA OOM or system memory exhaustion

**Solutions**:
1. Reduce batch size:
   ```yaml
   # In configs/minimal_reproduce.yaml
   performance:
     batch_size: 16  # Reduced from 32
   ```

2. Disable Jacobian computation:
   ```yaml
   performance:
     use_jacobian: false
   ```

3. Use CPU-only mode:
   ```bash
   CUDA_VISIBLE_DEVICES="" python scripts/run_reproduction.py ...
   ```

### Issue 3: Slow Execution

**Symptoms**: Reproduction takes >2x expected time

**Solutions**:
1. Enable quick mode:
   ```yaml
   performance:
     quick_mode: true
     skip_expensive_checks: true
   ```

2. Reduce sample count:
   ```yaml
   challenges:
     samples_per_family: 5  # Reduced from 10
   ```

3. Use parallel processing:
   ```yaml
   performance:
     parallel_workers: 4  # Increase if CPU allows
   ```

### Issue 4: Import Errors

**Symptoms**: ModuleNotFoundError or ImportError

**Solutions**:
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Verify Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check module installation
python -c "from pot.experiments import *; print('All modules imported')"
```

### Issue 5: Non-Deterministic Results

**Symptoms**: Different results across runs

**Solutions**:
```python
# In your script, add:
import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)
```

## Known Limitations

### Framework Limitations

1. **Model Size**: Large models (>10GB) may require memory optimization
2. **GPU Compatibility**: Tested primarily on NVIDIA GPUs (A100, V100)
3. **Tokenizer Variations**: LM results may vary with different tokenizer versions
4. **Numerical Precision**: Small variations (±0.001) expected across hardware

### Experimental Limitations

1. **Challenge Reuse**: Performance degrades with >25% challenge leakage
2. **Distribution Shift**: Calibration assumes stationary distributions
3. **Black-box Only**: Cannot detect weight-level manipulations
4. **API Limitations**: External APIs may have rate limits or non-determinism

### Expected Variations

Results may vary due to:
- Hardware differences (CPU vs GPU, GPU models)
- Library versions (PyTorch, NumPy)
- Operating system (Linux vs macOS vs Windows)
- Floating-point precision modes

Acceptable variation ranges:
- FAR/FRR: ±0.002 absolute
- Accuracy: ±0.01 absolute
- AUROC: ±0.01 absolute
- Query count: ±20% relative

## Advanced Reproduction Scenarios

### Custom Model Testing

Test your own models:

```python
from pot.security.proof_of_training import ProofOfTraining

# Load your model
import torch
model = torch.load('your_model.pth')

# Configure verification
config = {
    'verification_type': 'fuzzy',
    'model_type': 'vision',
    'security_level': 'high'
}

pot = ProofOfTraining(config)
result = pot.perform_verification(model, "custom_model", "standard")
print(f"Verification: {'PASSED' if result.verified else 'FAILED'}")
```

### Cross-Domain Verification

Test vision models on language tasks or vice versa:

```bash
# Vision model on mixed challenges
python scripts/run_verify.py \
    --config configs/vision_cifar10.yaml \
    --challenge_family "vision:freq,vision:texture,lm:templates" \
    --cross-domain

# Language model on vision-inspired challenges
python scripts/run_verify.py \
    --config configs/lm_small.yaml \
    --challenge_family "lm:visual_descriptions" \
    --n 256
```

### Ablation Studies

Run specific ablations:

```bash
# Sequential method comparison
for method in eb sprt hoeffding fixed; do
    python scripts/run_verify.py \
        --config configs/vision_cifar10.yaml \
        --seq $method \
        --output-suffix _$method
done

# Threshold sensitivity
for tau in 0.01 0.05 0.1; do
    python scripts/run_verify_enhanced.py \
        --config configs/vision_cifar10.yaml \
        --tau-id $tau \
        --output-suffix _tau$tau
done
```

### Performance Profiling

Profile reproduction performance:

```bash
# Enable profiling
python scripts/run_reproduction.py \
    --config configs/minimal_reproduce.yaml \
    --profile \
    --profile-output profile_results.json

# Analyze profile
python scripts/analyze_profile.py profile_results.json
```

## Getting Help

### Documentation

- **Framework Overview**: [CLAUDE.md](CLAUDE.md)
- **API Reference**: Run `python -c "from pot.experiments import ReportGenerator; help(ReportGenerator)"`
- **Examples**: See `examples/` directory
- **Tests**: Review `tests/test_reproducibility.py` for usage patterns

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Save intermediate results:
```yaml
# In config file
output:
  save_raw: true
  save_intermediate: true
  debug_mode: true
```

### Support

- **Issues**: Report at https://github.com/yourusername/PoT_Experiments/issues
- **Discussions**: Join at https://github.com/yourusername/PoT_Experiments/discussions
- **Email**: pot-support@example.com

## Conclusion

This guide should enable successful reproduction of the PoT experimental results. The framework is designed to be robust and reproducible across different environments. If you encounter issues not covered here, please consult the troubleshooting section or open an issue on GitHub.

Remember that perfect reproduction may not always be possible due to hardware and software variations, but results should fall within the specified tolerance ranges. The automated validation and reconciliation tools will help identify and explain any discrepancies.

Happy reproducing!