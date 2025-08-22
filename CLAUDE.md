# PoT Framework - Critical Instructions for Claude

## Overview
**Proof-of-Training (PoT)** - Cryptographic verification framework for neural network training integrity with black-box behavioral verification and zero-knowledge proof generation.

**Working Directory**: `/Users/rohanvinaik/PoT_Experiments`  
**Model Directory**: `/Users/rohanvinaik/LLM_Models`  
**Python**: 3.11.8  
**Rust**: 1.88.0+ (for ZK circuits)

## Core Architecture

```
PoT_Experiments/
├── src/pot/                 # Main framework
│   ├── core/               # Statistical verification & challenges
│   ├── security/           # Cryptographic protocols (TLSH, SSDEEP, Merkle)
│   ├── zk/                 # Zero-knowledge proofs (Halo2)
│   ├── lm/                 # Language model verification
│   └── vision/             # Vision model verification
├── scripts/                # All executable scripts
│   ├── run_all.sh         # Main validation pipeline
│   └── run_e2e_validation.py  # E2E pipeline (recommended)
├── manifests/              # Experiment YAML configs
├── experimental_results/   # Test outputs
└── data/                   # Data files
```

## Primary Validation Methods

### 1. E2E Pipeline (Recommended)
```bash
python scripts/run_e2e_validation.py \
    --ref-model gpt2 \
    --cand-model distilgpt2 \
    --mode audit
```

### 2. Full Validation Suite
```bash
bash scripts/run_all.sh                    # Full validation (15-20 min)
bash scripts/run_all.sh --skip-zk         # Without ZK proofs (5-10 min)
```

### 3. Manifest-Driven Pipeline
```bash
bash scripts/run_all.sh manifests/neurips_demo.yaml
python tools/pot_runner.py run --manifest manifests/neurips_demo.yaml --id exp_002
```

### 4. Model Selection Pipeline
```bash
python scripts/run_pipeline_with_models.py --models-dir /Users/rohanvinaik/LLM_Models
python scripts/run_pipeline_with_models.py --auto-pairs small --test-mode enhanced --non-interactive
```

## Enhanced Diff Decision Framework

**Location**: `src/pot/core/diff_decision.py`

### Key Classes
- `EnhancedSequentialTester` - Separate SAME/DIFFERENT decision logic
- `TestingMode.QUICK_GATE` - Fast checks (97.5% confidence, n_max=120)
- `TestingMode.AUDIT_GRADE` - High precision (99% confidence, n_max=400)

### Decision Rules
- **SAME**: CI within [-γ, +γ] AND half_width ≤ η·γ
- **DIFFERENT**: Effect size ≥ δ* AND RME ≤ ε_diff
- **UNDECIDED**: Provides diagnostics and suggestions

## Model Guidelines

### Open Models (No Auth Required)
- `gpt2`, `distilgpt2`, `gpt2-medium`
- `EleutherAI/pythia-70m`, `EleutherAI/pythia-160m`
- `microsoft/DialoGPT-medium`

### Gated Models (Require HF Login)
- `meta-llama/*`, `mistralai/*` (require authentication)

### Quick Model Download
```bash
# Open models
huggingface-cli download gpt2 --local-dir /Users/rohanvinaik/LLM_Models/gpt2 --local-dir-use-symlinks False

# For gated models (after huggingface-cli login)
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir /Users/rohanvinaik/LLM_Models/llama-2-7b-chat-hf --local-dir-use-symlinks False
```

## CRITICAL: Never Create Mock Tests

When asked for tests or Google Colab code:

1. **USE ACTUAL FRAMEWORK** - Never create simplified/mock versions
   - Statistical verification: `src.pot.core.diff_decision.EnhancedSequentialTester`
   - Fuzzy hashing: Real TLSH/SSDEEP algorithms
   - Provenance: Actual Merkle trees

2. **TESTS MUST BE COMPREHENSIVE**
   - Should take minutes, not seconds
   - Generate real metrics and confidence intervals
   - Save results to `experimental_results/`
   - Use actual PoT framework classes

3. **FOR GOOGLE COLAB**
   ```python
   # Clone repo, install deps, run actual pipeline
   !git clone <repo>
   !pip install torch transformers numpy scipy scikit-learn
   !python scripts/run_e2e_validation.py --dry-run
   ```

## Quick Troubleshooting

### Pre-flight Checks
```bash
pwd                          # Should be /Users/rohanvinaik/PoT_Experiments
python --version            # Should be 3.11.8
python -c "import torch, transformers, numpy, scipy; print('✅ Core deps OK')"
ls -la /Users/rohanvinaik/LLM_Models/  # Verify models exist
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pot` | Run from `/Users/rohanvinaik/PoT_Experiments` |
| `Permission denied` | `chmod +x scripts/*.sh scripts/*.py` |
| ZK tests fail | Use `--skip-zk` flag or install Rust |
| Out of memory | Use smaller models or `--skip-zk` |
| Models not found | Use open models (gpt2, distilgpt2) |

### Emergency Reset
```bash
pkill -f python && pkill -f cargo
rm -rf /tmp/pot_* experimental_results/temp_*
cd /Users/rohanvinaik/PoT_Experiments
bash scripts/run_all.sh --skip-zk
```

## Expected Results

- **Deterministic Tests**: 100% success rate
- **Statistical Tests**: >95% success for clear cases
- **Security Tests**: 80%+ accuracy for fuzzy hashing
- **ZK Tests**: Health score >70/100
- **Performance**: 
  - Small models: 1-2 sec/query
  - Medium (Pythia): ~1 sec/query
  - Large (7B+): 8-10 sec/query

## Dependencies

```bash
# Required
pip install torch>=2.2.0 transformers>=4.36.2 numpy scipy scikit-learn

# Optional
pip install tlsh  # Fuzzy hashing (SSDeep warnings are normal)
```

## Key Scripts Reference

- `run_e2e_validation.py` - Complete validation with reporting
- `run_enhanced_diff_test.py` - Statistical verification
- `test_size_fraud_detection.py` - Size fraud detection
- `run_pipeline_with_models.py` - Custom model testing
- `run_zk_validation.py` - ZK proof validation

## Creating Custom Manifests

```yaml
# manifests/my_experiment.yaml
run_id: "my-experiment"
hmac_key_hex: "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f"

global:
  n_challenges: 32
  out_root: "runs/my_experiment"
  mode: "AUDIT"  # QUICK, AUDIT, or EXTENDED

experiments:
  - id: exp_001
    ref:
      type: hf_local
      model_path: /Users/rohanvinaik/LLM_Models/gpt2
    cand:
      type: hf_local
      model_path: /Users/rohanvinaik/LLM_Models/distilgpt2
```

## Remember
This framework validates academic paper claims. **Never create mock tests** - always use the real PoT framework code. Tests should be comprehensive and take several minutes to run, generating real metrics and evidence bundles.
