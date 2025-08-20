# 📚 Model Selection Guide

## Overview

The ZK-PoT framework now supports flexible model selection from multiple sources:
- **Local filesystem**: Load models from your local disk
- **HuggingFace Hub**: Download and use models directly from HuggingFace
- **Auto mode**: Automatically try local first, then fallback to HuggingFace

## Quick Start

### 1. Basic Usage

```bash
# Use local models (default searches in common directories)
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a gpt2 \
  --model-b distilgpt2 \
  --model-source local

# Use HuggingFace models
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a gpt2 \
  --model-b distilgpt2 \
  --model-source huggingface

# Auto mode (try local first, then HuggingFace)
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a mistral \
  --model-b zephyr \
  --model-source auto
```

### 2. Using Configuration Files

```bash
# Load from predefined config
python scripts/runtime_blackbox_validation_configurable.py \
  --config configs/validation_examples.json

# Save your configuration for reuse
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a gpt2 \
  --model-b distilgpt2 \
  --save-config my_config.json
```

## Model Sources

### Local Models

The framework automatically searches for models in these locations:
1. `/Users/rohanvinaik/LLM_Models` (default)
2. `~/LLM_Models`
3. `~/models`
4. `./models`
5. `../models`

You can specify a custom location:
```bash
python scripts/runtime_blackbox_validation_configurable.py \
  --local-model-base /path/to/your/models \
  --model-source local
```

### HuggingFace Hub

Access thousands of models from HuggingFace:

```bash
# Public models (no authentication needed)
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a EleutherAI/pythia-70m \
  --model-b EleutherAI/pythia-1.4b \
  --model-source huggingface

# Private/gated models (requires HF_TOKEN)
export HF_TOKEN=your_huggingface_token
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a meta-llama/Llama-2-7b-hf \
  --model-b meta-llama/Llama-2-7b-chat-hf \
  --model-source huggingface \
  --use-hf-token
```

## Model Name Mappings

The framework understands common model aliases:

| Alias | Local Path | HuggingFace ID |
|-------|------------|----------------|
| `gpt2` | `gpt2/` | `gpt2` |
| `distilgpt2` | `distilgpt2/` | `distilgpt2` |
| `mistral` | `mistral_for_colab/` | `mistralai/Mistral-7B-v0.1` |
| `zephyr` | `zephyr-7b-beta-final/` | `HuggingFaceH4/zephyr-7b-beta` |
| `llama2-7b` | `llama-2-7b-hf/` | `meta-llama/Llama-2-7b-hf` |
| `llama2-7b-chat` | `llama-2-7b-chat-hf/` | `meta-llama/Llama-2-7b-chat-hf` |
| `falcon-7b` | `falcon-7b/` | `tiiuae/falcon-7b` |
| `vicuna-7b` | `vicuna-7b-v1.5/` | `lmsys/vicuna-7b-v1.5` |
| `phi-2` | `phi-2/` | `microsoft/phi-2` |

## Memory Optimization

### For Large Models (7B+)

Use FP16 precision to reduce memory usage by 50%:

```bash
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a mistral \
  --model-b zephyr \
  --torch-dtype float16 \
  --device-map auto
```

### For Limited Memory

Reduce batch size and query count:

```bash
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a gpt2 \
  --model-b distilgpt2 \
  --n-queries 8 \
  --positions-per-prompt 16
```

## Test Modes

### Adaptive Mode (Default)
Dynamically adjusts sampling based on convergence:
```bash
--test-mode adaptive --n-queries 12
```

### Quick Gate Mode
Fast initial verification (97.5% confidence):
```bash
--test-mode quick_gate --confidence 0.975
```

### Audit Grade Mode
High-precision verification (99% confidence):
```bash
--test-mode audit_grade --confidence 0.99 --n-queries 30
```

## Examples

### 1. Compare Small Local Models
```bash
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a gpt2 \
  --model-b distilgpt2 \
  --model-source local \
  --test-mode adaptive
```

### 2. Compare Large HuggingFace Models
```bash
export HF_TOKEN=your_token
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a mistralai/Mistral-7B-v0.1 \
  --model-b HuggingFaceH4/zephyr-7b-beta \
  --model-source huggingface \
  --torch-dtype float16 \
  --use-hf-token
```

### 3. Cross-Size Comparison
```bash
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a gpt2 \
  --model-b mistral \
  --model-source auto \
  --torch-dtype float16 \
  --positions-per-prompt 16
```

### 4. Base vs Fine-tuned
```bash
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a llama2-7b \
  --model-b llama2-7b-chat \
  --model-source auto \
  --test-mode audit_grade \
  --n-queries 30
```

## Environment Variables

- `HF_TOKEN`: HuggingFace authentication token
- `LOCAL_MODEL_BASE`: Default directory for local models
- `HF_HOME`: HuggingFace cache directory

## Troubleshooting

### Out of Memory
- Use `--torch-dtype float16` for large models
- Reduce `--n-queries` and `--positions-per-prompt`
- Use `--device-map auto` for automatic device allocation

### Model Not Found
- Check model name/path is correct
- For local models, ensure `config.json` exists in model directory
- For HuggingFace, check model ID is correct

### Authentication Error
- Set `HF_TOKEN` environment variable
- Use `--use-hf-token` flag
- Check token has access to requested models

### Slow Loading
- First-time HuggingFace downloads can be slow
- Use local models for faster loading
- Consider using `--hf-cache-dir` for persistent cache

## Python API

```python
from pot.core.model_loader import UnifiedModelLoader, ModelSource

# Create loader
loader = UnifiedModelLoader(
    local_base="/path/to/models",
    default_source=ModelSource.AUTO
)

# Load models
model_a, tokenizer_a = loader.load("gpt2")
model_b, tokenizer_b = loader.load("distilgpt2")

# List available models
available = loader.list_available_models()
print(f"Local models: {available['local']}")
print(f"HF models: {available['huggingface']}")
```

## Configuration File Format

```json
{
  "model_a": "gpt2",
  "model_b": "distilgpt2",
  "model_source": "auto",
  "local_model_base": "/Users/rohanvinaik/LLM_Models",
  "hf_cache_dir": "~/.cache/huggingface",
  "use_hf_token": false,
  "torch_dtype": "auto",
  "device_map": null,
  "trust_remote_code": false,
  "test_mode": "adaptive",
  "n_queries": 12,
  "positions_per_prompt": 32,
  "confidence": 0.95,
  "gamma": 0.02,
  "delta_star": 0.08,
  "epsilon_diff": 0.25,
  "output_results": true,
  "output_dir": "experimental_results",
  "verbose": false
}
```

## Advanced Features

### Custom Model Paths
```bash
# Direct path to model directory
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a /absolute/path/to/model1 \
  --model-b /absolute/path/to/model2 \
  --model-source local
```

### Specific HuggingFace Revisions
```python
# In code, you can specify exact commits/branches
config = ModelConfig(
    name="gpt2",
    source=ModelSource.HUGGINGFACE,
    revision="main"  # or specific commit hash
)
```

### Multi-GPU Support
```bash
# Automatic device mapping for large models
python scripts/runtime_blackbox_validation_configurable.py \
  --model-a mistral \
  --model-b zephyr \
  --device-map auto \
  --torch-dtype float16
```

## Performance Tips

1. **Use local models** when possible for faster loading
2. **Enable FP16** for large models to save memory
3. **Adjust batch sizes** based on available memory
4. **Use adaptive mode** for efficient sampling
5. **Cache HuggingFace models** to avoid re-downloading

## Support Matrix

| Model Size | Local Support | HF Support | Recommended dtype |
|------------|--------------|------------|-------------------|
| <1B params | ✅ | ✅ | float32 |
| 1-7B params | ✅ | ✅ | float16 |
| 7-13B params | ✅ | ✅ | float16 + device_map |
| >13B params | ⚠️ | ⚠️ | bfloat16 + multi-GPU |

## Contributing

To add support for new model sources or formats:
1. Extend `ModelSource` enum in `pot/core/model_loader.py`
2. Add mappings to `UnifiedModelLoader`
3. Update this documentation
4. Submit a pull request