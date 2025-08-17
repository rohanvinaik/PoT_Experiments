# PoT Model Setup System

A comprehensive model management system for the Proof-of-Training framework that provides automatic downloading, caching, and fallback mechanisms for reproducible experiments.

## 🏗️ Overview

The `MinimalModelSetup` system provides:
- **Automatic Model Downloads**: HuggingFace and torchvision integration
- **Intelligent Caching**: Local storage with checksum verification  
- **Configuration Presets**: Minimal, test, and paper configurations
- **Robust Fallbacks**: Mock models when downloads fail
- **Memory Optimization**: Usage tracking and cache management
- **Full Reproducibility**: Deterministic model loading with seeds

## 📦 Quick Start

```python
from pot.experiments.model_setup import MinimalModelSetup, ModelConfig, ModelPreset

# Initialize setup system
setup = MinimalModelSetup()

# Load models with different presets
vision_config = ModelConfig("vision_model", "vision", ModelPreset.MINIMAL)
vision_model = setup.get_vision_model(vision_config)

language_config = ModelConfig("language_model", "language", ModelPreset.MINIMAL) 
language_model = setup.get_language_model(language_config)

# Convenience functions
from pot.experiments.model_setup import get_minimal_vision_model, get_test_models

vision = get_minimal_vision_model()
test_models = get_test_models()
```

## 🎯 Configuration Presets

### TEST - Mock Models for CI/CD
- **Memory**: <10MB total
- **Purpose**: Fast testing, continuous integration
- **Models**: Simple PyTorch architectures
- **Features**: Instant loading, deterministic behavior

```python
# Vision: Simple CNN (3→32→10 conv layers)
# Language: Basic LSTM with mock tokenizer
test_models = get_test_models()
```

### MINIMAL - Lightweight Production Models  
- **Memory**: 50-250MB
- **Purpose**: Resource-constrained deployments
- **Models**: MobileNetV2, DistilBERT
- **Features**: Real pretrained weights, good performance

```python
# Vision: MobileNetV2 (13.5MB actual usage)
# Language: DistilBERT (253MB actual usage)
vision = get_minimal_vision_model()
language = get_minimal_language_model()
```

### PAPER - Research-Grade Models
- **Memory**: 200-500MB  
- **Purpose**: Academic research, full experiments
- **Models**: ResNet18, BERT-base
- **Features**: Standard research baselines

```python
# Vision: ResNet18 (44.6MB actual usage)
# Language: BERT-base (417MB actual usage)
paper_models = get_paper_models()
```

## 🔧 Model Specifications

| Model | Preset | Source | Memory | Description |
|-------|--------|--------|--------|-------------|
| **Vision Models** |
| Mock CNN | TEST | Mock | 1.0MB | Simple conv layers for testing |
| MobileNetV2 | MINIMAL | torchvision | 13.5MB | Efficient mobile vision |
| ResNet18 | PAPER | torchvision | 44.6MB | Standard research baseline |
| **Language Models** |
| Mock Transformer | TEST | Mock | 1.7MB | Basic LSTM with tokenizer |
| DistilBERT | MINIMAL | HuggingFace | 253MB | Compressed BERT variant |
| BERT-base | PAPER | HuggingFace | 417MB | Standard transformer baseline |

## 💾 Caching System

Models are cached in `~/.cache/pot_experiments/` with:
- **Automatic Downloads**: First access downloads and caches
- **Checksum Verification**: SHA256 hashes for reproducibility
- **Smart Reuse**: Identical configurations reuse cached models
- **Memory Management**: Clear cache to free memory

```python
setup = MinimalModelSetup()

# First load downloads model
model1 = setup.get_vision_model(config)  # ~1s download

# Second load uses cache  
model2 = setup.get_vision_model(config)  # ~0.0001s cached

# Memory management
memory_report = setup.get_memory_report()
setup.clear_cache()  # Free memory
```

## 🔄 Fallback Mechanism

Robust error handling ensures experiments always run:

```python
def get_model_with_fallback(model_type: str, preset: str = "minimal"):
    try:
        return download_real_model(model_type, preset)
    except Exception as e:
        logger.warning(f"Using mock for {model_type}: {e}")
        return get_mock_model(model_type)
```

**Fallback Chain**:
1. **Try Preset Model**: Download requested model
2. **Try Cached Version**: Use local cache if available  
3. **Fallback to Mock**: Create simple mock model
4. **Never Fail**: Always returns working model

## 🧪 Integration with Experiments

The model setup integrates seamlessly with `ReproducibleExperimentRunner`:

```python
from pot.experiments.reproducible_runner import ReproducibleExperimentRunner, ExperimentConfig

config = ExperimentConfig(
    experiment_name="vision_experiment",
    model_type="vision",
    model_architecture="minimal",  # Uses MobileNetV2
    # ... other config
)

runner = ReproducibleExperimentRunner(config)
runner.setup_models()  # Automatically uses MinimalModelSetup
```

**Architecture Mapping**:
- `"mock"`, `"test"` → TEST preset
- `"minimal"`, `"mobilenet"`, `"distilbert"` → MINIMAL preset  
- `"paper"`, `"resnet18"`, `"bert"` → PAPER preset

## 🔒 Reproducibility Features

Ensures deterministic model loading:

```python
# Seed management
torch.manual_seed(config.seed)
model = load_model()  # Deterministic initialization

# Checksum verification  
expected_hash = "abc123..."
assert verify_checksum(model, expected_hash)

# Model registry
setup.save_model_registry("models.json")  # Audit trail
```

## 📊 Memory Usage Monitoring

Track and optimize memory usage:

```python
setup = MinimalModelSetup()

# Load multiple models
vision = setup.get_vision_model(config)
language = setup.get_language_model(config)

# Memory analysis
report = setup.get_memory_report()
print(f"Total: {report['total_memory_mb']:.1f}MB")
print(f"Largest: {report['largest_model']}")

# Optimization
setup.clear_cache()  # Free memory when done
```

## 🛠️ Advanced Usage

### Custom Model Configuration

```python
config = ModelConfig(
    name="custom_vision",
    model_type="vision", 
    preset=ModelPreset.MINIMAL,
    device="cuda",
    seed=42,
    verify_checksum=True,
    fallback_to_mock=True,
    memory_limit_mb=100
)

model_info = setup.get_vision_model(config)
```

### Model Information Access

```python
model_info = setup.get_vision_model(config)

print(f"Source: {model_info.source}")        # "pretrained", "mock", "cached"
print(f"Memory: {model_info.memory_mb}MB")   # Actual memory usage  
print(f"Config: {model_info.config}")       # Model-specific config
print(f"Checksum: {model_info.checksum}")   # SHA256 hash

# PyTorch model
model = model_info.model
output = model(input_tensor)

# Tokenizer (language models only)
if model_info.tokenizer:
    tokens = model_info.tokenizer.encode("Hello world")
```

### Availability Checking

```python
available = setup.list_available_models()

for model_type, presets in available.items():
    for preset, spec in presets.items():
        if spec["available"]:
            print(f"✅ {model_type}/{preset}: {spec['description']}")
        else:
            print(f"❌ {model_type}/{preset}: Not available")
```

## 🧪 Testing

Comprehensive test suite ensures reliability:

```bash
# Test all functionality
python test_model_setup.py

# Demo the system
python example_model_setup_demo.py
```

**Test Coverage**:
- ✅ Mock model creation
- ✅ Vision model loading (all presets)
- ✅ Language model loading (all presets)  
- ✅ Fallback mechanisms
- ✅ Caching and performance
- ✅ Memory management
- ✅ Integration with experiments

## 🎯 Best Practices

### For Development
```python
# Use test models for fast iteration
test_models = get_test_models()
```

### For Production
```python
# Use minimal models for efficiency
vision = get_minimal_vision_model()
language = get_minimal_language_model()
```

### For Research
```python
# Use paper models for full experiments
paper_models = get_paper_models()
```

### For CI/CD
```python
# Always use fallbacks in automated environments
config.fallback_to_mock = True
```

## 🔍 Troubleshooting

### Common Issues

**Download Failures**:
```python
# Enable fallback for reliability
config.fallback_to_mock = True
```

**Memory Issues**:
```python
# Clear cache periodically
setup.clear_cache()

# Use smaller models
config.preset = ModelPreset.MINIMAL
```

**Reproducibility Issues**:
```python
# Always set seeds
config.seed = 42

# Verify checksums
config.verify_checksum = True
```

### Error Messages

- `"Transformers not available"` → Install: `pip install transformers`
- `"Torchvision not available"` → Install: `pip install torchvision`
- `"Using mock for X"` → Normal fallback behavior
- `"Checksum mismatch"` → Model weights differ, check version

## 📁 File Structure

```
pot/experiments/
├── model_setup.py              # Main implementation
├── README_model_setup.md       # This documentation
├── reproducible_runner.py      # Integration with experiments
└── __init__.py

tests/
├── test_model_setup.py         # Comprehensive tests
└── example_model_setup_demo.py # Usage demonstrations

cache/
└── ~/.cache/pot_experiments/   # Model cache directory
    ├── models/                 # Downloaded models
    └── checksums.json         # Verification data
```

## 🔗 Related Documentation

- [`reproducible_runner.py`](reproducible_runner.py) - Experiment integration
- [`../core/challenge.py`](../core/challenge.py) - Challenge generation
- [`../vision/verifier.py`](../vision/verifier.py) - Vision verification
- [`../lm/verifier.py`](../lm/verifier.py) - Language verification

---

## 🏆 Key Features Summary

✅ **Automatic Downloads**: HuggingFace & torchvision integration  
✅ **Smart Caching**: Local storage with checksum verification  
✅ **Three Presets**: Test, minimal, and paper configurations  
✅ **Robust Fallbacks**: Never fails with mock model support  
✅ **Memory Tracking**: Usage monitoring and optimization  
✅ **Full Integration**: Works with reproducible experiment runner  
✅ **Reproducible**: Deterministic loading with seed management  
✅ **Well Tested**: Comprehensive test suite and examples