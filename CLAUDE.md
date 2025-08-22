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

### 1. Unified E2E Pipeline (Recommended)
```bash
# Basic validation with all integrated features
python scripts/run_e2e_validation.py \
    --ref-model gpt2 \
    --cand-model distilgpt2 \
    --mode audit

# Enhanced validation with all CI/CD components
python scripts/run_e2e_validation.py \
    --ref-model gpt2 \
    --cand-model distilgpt2 \
    --mode audit \
    --enable-attack-simulation \
    --enable-sharding \
    --performance-dashboard \
    --test-data-generation
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
| CI/CD components warnings | Normal if `tests.fixtures` not installed - core PoT still works |

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
  - Small models (GPT-2, DistilGPT-2): ~2.2 sec/query
  - Medium (Pythia): ~1 sec/query
  - Large (7B+): 8-10 sec/query
- **Self-consistency (SAME model)**: 
  - Audit mode: ~30 queries, 99% confidence
  - Decision time: ~65 seconds for GPT-2
- **Distillation detection (DIFFERENT models)**:
  - GPT-2 vs DistilGPT-2: ~32 queries, effect size ~13

## Dependencies

```bash
# Required
pip install torch>=2.2.0 transformers>=4.36.2 numpy scipy scikit-learn

# Optional
pip install tlsh  # Fuzzy hashing (SSDeep warnings are normal)
```

## Key Scripts Reference

- `run_e2e_validation.py` - **Unified E2E pipeline with all CI/CD features** (recommended)
- `run_memory_safe_validation.py` - **Memory-safe runner for large models (7B+)**
- `test_7b_models_safe.py` - Test suite for 7B model permutations
- `run_enhanced_diff_test.py` - Statistical verification (legacy)
- `test_size_fraud_detection.py` - Size fraud detection
- `run_pipeline_with_models.py` - Custom model testing
- `run_zk_validation.py` - ZK proof validation

### Unified E2E Pipeline Features
The main `run_e2e_validation.py` script now includes:
- **Performance tracking** with SQLite database and real-time monitoring
- **Audit trail validation** with cryptographic hash chain verification
- **Adversarial attack simulation** (replay, timing, injection attacks)
- **Model sharding** for large models (34B+ parameters)
- **Evidence bundle generation** with cryptographic signing
- **Performance dashboards** with metrics visualization
- **Test data management** for CI testing environments
- **Automatic README table updates** with rolling metrics tracking
- **HTML report generation** for human-readable validation results
- **Memory-safe execution** with configurable limits (--max-memory-percent)
- **Sequential processing** for large models (--enforce-sequential)

## Memory-Safe Validation for Large Models (7B+)

### Problem
Large language models (7B+ parameters) can consume excessive memory, causing OOM errors when running multiple tests in parallel or without proper memory management.

### Solution
The framework now includes dedicated memory-safe validation infrastructure:

#### 1. Memory-Safe Runner (`run_memory_safe_validation.py`)
```bash
# Run 7B model permutations with 25% memory limit
python scripts/run_memory_safe_validation.py \
    --models yi-6b yi-34b \
    --permutations all \
    --max-memory 25

# Custom thresholds
python scripts/run_memory_safe_validation.py \
    --ref-model meta-llama/Llama-2-7b-hf \
    --cand-model meta-llama/Llama-2-7b-chat-hf \
    --max-memory 25 \
    --sequential-threshold 5.0 \
    --sharding-threshold 10.0
```

**Features:**
- **25% memory limit enforcement** (configurable)
- **Sequential execution** for models >5GB
- **Automatic sharding** for models >10GB
- **3x retry with memory cleanup** on failures
- **Real-time memory monitoring**
- **Checkpoint/recovery support**

#### 2. Enhanced E2E Pipeline
```bash
# Run with strict memory limits
python scripts/run_e2e_validation.py \
    --ref-model yi-6b \
    --cand-model yi-34b \
    --mode audit \
    --enable-sharding \
    --max-memory-percent 25 \
    --enforce-sequential
```

#### 3. 7B Model Test Suite (`test_7b_models_safe.py`)
```bash
# Run the standard 3 permutations (A|A, B|B, A|B)
python scripts/test_7b_models_safe.py

# With memory-safe runner
python scripts/test_7b_models_safe.py --memory-safe
```

**Test sequence:**
1. Model A self-consistency (A|A)
2. 30-second cooldown
3. Model B self-consistency (B|B) 
4. 30-second cooldown
5. Cross-model comparison (A|B)

### Memory Management Strategy
- **Small models (<1GB)**: Can run in parallel
- **Medium models (1-5GB)**: Sequential recommended
- **Large models (5-20GB)**: Sequential required, sharding recommended
- **XLarge models (>20GB)**: Sequential required, sharding required

### Error Recovery
1. **Automatic retry**: Up to 3 attempts with cleanup
2. **Memory cleanup**: Forced GC between tests
3. **Timeout protection**: 30-minute limit per test
4. **Graceful degradation**: Falls back to simpler verification if needed

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

## CRITICAL MISTAKES TO AVOID ("Sin Bucket")

### 1. NEVER Use Fixed Query Limits
**WRONG:**
```bash
python scripts/run_e2e_validation.py --max-queries 10  # WRONG - defeats sequential testing
python scripts/run_e2e_validation.py --max-queries 50  # WRONG - hardcoded limit
```

**CORRECT:**
```bash
python scripts/run_e2e_validation.py --mode quick    # Let framework decide queries dynamically
python scripts/run_e2e_validation.py --mode audit    # Framework uses statistical confidence
```

The entire point of the sequential testing framework is to **dynamically determine** the number of queries needed based on statistical confidence. The framework will:
- Stop early (10-30 queries) when models are clearly SAME or DIFFERENT
- Continue longer (up to mode limits) when confidence is insufficient
- Use mode-specific limits: QUICK (120), AUDIT (400), EXTENDED (800)

**DO NOT** override this with --max-queries unless explicitly testing edge cases.

### 2. NEVER Create Mock/Simplified Tests
Always use the real PoT framework - tests should take minutes and generate real metrics.

### 3. NEVER Run Large Models in Parallel
Models >5GB must run sequentially with proper memory management.

## Remember
This framework validates academic paper claims. **Never create mock tests** - always use the real PoT framework code. Tests should be comprehensive and take several minutes to run, generating real metrics and evidence bundles.
