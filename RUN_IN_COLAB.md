# Running PoT Tests in Google Colab

## Quick Start (One-Click Setup)

1. **Open Google Colab**: https://colab.research.google.com/

2. **Create a new notebook** and paste this single cell:

```python
# One-click PoT Test Suite for Google Colab
!git clone https://github.com/rohanvinaik/PoT_Experiments.git
!cd PoT_Experiments && python colab_all_pot_tests.py
```

That's it! The notebook will:
- Clone the repository
- Install all dependencies
- Run the complete test suite
- Generate a comprehensive report

## Using the Full Notebook

For more control and visibility, you can use the complete notebook:

1. **Download**: `colab_all_pot_tests.py` from this repository
2. **Upload to Colab**: File → Upload notebook
3. **Run all cells**: Runtime → Run all

## What Gets Tested

The notebook runs the complete PoT validation suite:

### Core Components
- ✅ **Statistical Difference Framework** - Model verification with confidence intervals
- ✅ **Fuzzy Hash Verifier** - Behavioral fingerprinting
- ✅ **Training Provenance Auditor** - Merkle tree validation
- ✅ **Token Space Normalizer** - Alignment scoring
- ✅ **PRF Module** - Deterministic key derivation
- ✅ **Boundaries Module** - Sequential testing

### Validation Tests
- ✅ **Deterministic Validation** - Consistent testing with mock models
- ✅ **Challenge Generation** - Vision, language, and generic challenges
- ✅ **Batch Processing** - Multi-model verification
- ✅ **Stress Tests** - Performance with large-scale data
- ✅ **LLM Verification** - Testing with real language models (GPT-2)

### Performance Benchmarks
- Throughput testing (>4000 verifications/second)
- Memory efficiency validation
- Batch processing optimization
- Large challenge vector handling (up to 25,000 dimensions)

## Results

After running, you'll find:
- **Comprehensive Report**: `colab_test_results/comprehensive_report_*.txt`
- **Detailed JSON Results**: `colab_test_results/deterministic_validation_*.json`
- **Test Logs**: Individual test outputs in `colab_test_results/`

## GPU Acceleration

The notebook automatically detects and uses GPU if available:
- Speeds up model inference
- Enables testing with larger models
- Improves batch processing performance

To ensure GPU is enabled:
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Save and run

## Troubleshooting

### "Module not found" errors
The notebook automatically installs all dependencies. If you see import errors:
```python
!pip install torch transformers numpy scipy matplotlib pytest
```

### GPU not detected
Check Runtime → Change runtime type → Hardware accelerator is set to GPU

### Tests failing
This is often due to random model initialization. The deterministic tests should always pass.

## Advanced Usage

### Running Specific Tests

To run only the statistical difference framework tests:
```python
!cd PoT_Experiments && python -m pytest pot/core/test_diff_decision.py -v
```

To run only the stress tests:
```python
!cd PoT_Experiments && python examples/test_statistical_difference.py
```

### Customizing Test Configuration

Edit the configuration in the notebook:
```python
config = DiffDecisionConfig(
    n_min=20,           # Minimum samples
    n_max=100,          # Maximum samples  
    alpha=0.01,         # 99% confidence
    method="eb"         # Empirical-Bernstein bounds
)
```

## Expected Output

A successful run shows:
```
======================================================================
COMPREHENSIVE POT TEST SUITE - FINAL REPORT
======================================================================
...
🎯 Success Rate: 85%+

STATUS: ✅ READY FOR PRODUCTION
```

## Support

- **Issues**: https://github.com/rohanvinaik/PoT_Experiments/issues
- **Documentation**: See `/docs` folder in the repository
- **Statistical Framework**: `docs/STATISTICAL_DIFFERENCE_FRAMEWORK.md`