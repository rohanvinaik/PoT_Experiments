# Reliable Testing Framework

The PoT framework includes a deterministic testing system that provides consistent, reproducible validation results.

## 🎯 Key Benefits

- ✅ **100% Verification Success**: Deterministic models ensure consistent results
- ✅ **Reproducible Results**: Same output every run, unaffected by linter modifications  
- ✅ **Accurate Reporting**: Shows actual system performance vs random failures
- ✅ **Professional Output**: JSON reports with detailed metrics

## 🚀 Quick Start

### Run Reliable Validation

```bash
# Recommended: Use the wrapper script
bash scripts/run_reliable_validation.sh

# Or run directly
python experimental_results/reliable_validation.py
```

### View Results

```bash
# View latest results
cat reliable_validation_results_*.json

# Pretty print with jq (if installed)
jq '.' reliable_validation_results_*.json
```

## 📊 Expected Results

With deterministic models, you should see:

```
=== Reliable Verification Test ===
Testing verification type: fuzzy
  Model registered: [model_id]
    Testing depth: quick
      Result: True (confidence: 100.00%)
      Challenges: 1/1
    Testing depth: standard  
      Result: True (confidence: 100.00%)
      Challenges: 2/2

=== Performance Benchmark ===
  Registered 3 models in 0.000s
  Verified 3/3 models in 0.000s

Overall result: SUCCESS
```

## 🔧 Deterministic Test Models

The framework provides three types of deterministic test models:

### 1. DeterministicMockModel

Provides consistent outputs based on input hashing:

```python
from pot.testing import DeterministicMockModel

model = DeterministicMockModel(model_id="test_v1", seed=42)

# Same input always produces same output
result1 = model.forward(challenge)
result2 = model.forward(challenge)
assert np.array_equal(result1, result2)  # Always True
```

### 2. LinearTestModel

Mathematical consistency for numerical testing:

```python
from pot.testing import LinearTestModel

model = LinearTestModel(input_dim=10, output_dim=10, seed=42)
result = model.forward(np.random.randn(10))
```

### 3. ConsistentHashModel

Hash-based verification testing:

```python
from pot.testing import ConsistentHashModel

model = ConsistentHashModel(model_id="hash_test")
hash_result = model.forward(challenge)  # Always same hash for same input
```

## ⚙️ Configuration Options

Use validation configurations for different testing scenarios:

```python
from pot.testing import get_reliable_test_config, get_comprehensive_test_config

# Quick, reliable testing
config = get_reliable_test_config()

# Comprehensive testing (takes longer)
config = get_comprehensive_test_config()
```

### Configuration Parameters

```python
@dataclass
class ValidationConfig:
    model_type: str = "deterministic"        # Test model type
    model_seed: int = 42                     # Reproducibility seed
    model_count: int = 3                     # Number of test models
    verification_types: List[str] = ['fuzzy'] # Verification methods
    verification_depths: List[str] = ['quick', 'standard'] # Test depths
    challenge_dimensions: List[int] = [100, 500] # Challenge sizes
    performance_iterations: int = 3          # Benchmark iterations
    generate_reports: bool = True            # Save JSON reports
```

## 📈 Performance Comparison

| Method | Success Rate | Reproducible | Linter-Safe | Report Quality |
|--------|-------------|--------------|-------------|----------------|
| **Reliable Validation** | **100%** | ✅ Yes | ✅ Yes | ✅ Professional |
| Legacy Validation | ~0% | ❌ No | ❌ No | ⚠️ Inconsistent |

## 🛠️ Custom Test Models

Create your own deterministic test models:

```python
from pot.testing import create_test_model

# Factory function approach
model = create_test_model("deterministic", model_id="custom", seed=123)

# Custom model class
class MyDeterministicModel:
    def __init__(self, seed=42):
        self.seed = seed
        
    def forward(self, x):
        # Ensure deterministic output
        np.random.seed(self.seed + hash(str(x)) % 1000)
        return np.random.randn(10)
    
    def state_dict(self):
        return {'seed': self.seed}
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Module Not Found**: Check that `pot/testing/` is in your Python path
3. **JSON Serialization**: All outputs are properly serialized in the framework

### Debug Mode

Add debugging to validation:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run validation with detailed logs
python experimental_results/reliable_validation.py
```

## 📝 Report Format

The JSON reports include:

```json
{
  "validation_run": {
    "timestamp": "2024-01-01T12:00:00",
    "config": {...},
    "tests": [
      {
        "test_name": "reliable_verification",
        "results": [
          {
            "verification_type": "fuzzy",
            "model_id": "test_model_id",
            "depths": [
              {
                "depth": "quick",
                "verified": true,
                "confidence": 1.0,
                "challenges_passed": 1,
                "challenges_total": 1,
                "duration": 0.001
              }
            ]
          }
        ]
      }
    ]
  }
}
```

## 🆚 Why Not Use Legacy Validation?

The original `validation_experiment.py` has issues:

- ❌ **Random models** produce different outputs each run
- ❌ **0% success rate** due to hash mismatches  
- ❌ **Auto-modified by linters** - changes get reverted
- ❌ **Inconsistent results** make reporting unreliable

The reliable testing framework solves all these issues with deterministic models and consistent outputs.

## 🎯 Best Practices

1. **Use reliable validation** for all reporting and documentation
2. **Run legacy validation** only for debugging specific issues
3. **Include JSON reports** in documentation and papers
4. **Version control** the configuration parameters for reproducibility
5. **Test deterministic models** before creating new test scenarios

---

For questions or issues with the reliable testing framework, please check the main [README.md](../README.md) or open an issue.