# PoT Framework - Critical Instructions for Claude

## CRITICAL MISTAKES TO AVOID ("Sin Bucket") - READ THIS FIRST!

### 1. NEVER Add Timeouts to Experiments
**WRONG:**
```bash
timeout 1200 python scripts/run_e2e_validation.py ...  # WRONG - RUINS EXPERIMENTS
timeout 3600 python test.py  # WRONG - TRUNCATES RESULTS
```

**CORRECT:**
```bash
python scripts/run_e2e_validation.py ...  # Let it run to completion
# If you need to stop, use Ctrl+C or kill manually, don't use timeout
```

Timeouts have RUINED HOURS of experiments by truncating them. Tests need to run to completion for valid results.

### 2. NEVER Run Large/Multiple Tests Concurrently
**WRONG:**
```bash
# Running multiple huge tests at once - CAUSES OOM AND SYSTEM CRASHES
python test1.py & python test2.py & python test3.py &  # WRONG
parallel -j4 python test.py ::: model1 model2 model3 model4  # WRONG
```

**CORRECT:**
```bash
# Run tests sequentially, especially for large models
python scripts/run_memory_safe_validation.py --enforce-sequential
python test1.py && python test2.py && python test3.py  # Sequential execution
```

### 2. NEVER Create Simplified/Mock Tests When Encountering Issues
**WRONG:**
```python
# Creating a "simpler" test when the real test fails
def simple_mock_test():  # WRONG - defeats the purpose
    return {"result": "passed", "confidence": 0.99}
```

**CORRECT:**
```python
# Use the actual framework, fix the root cause
# If OOM: Use memory-safe runner with --max-memory 25
# If timeout: Increase timeout or use --skip-zk
# If missing deps: Install them, don't mock them
```

### 3. NEVER Use Fixed Query Limits
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

### 4. NEVER Create Mock/Simplified Tests
Always use the real PoT framework - tests should take minutes and generate real metrics.

### 5. NEVER Run Large Models in Parallel
Models >5GB must run sequentially with proper memory management.

## Overview
**Proof-of-Training (PoT)** - Cryptographic verification framework for neural network training integrity with black-box behavioral verification and zero-knowledge proof generation.

**Working Directory**: `~/PoT_Experiments`  
**Model Directory**: `~/LLM_Models`  
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
python scripts/run_pipeline_with_models.py --models-dir ~/LLM_Models
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
- **UNDECIDED_STABLE**: Behavioral fingerprint detected (stable intermediate state)

## Adaptive Variance Reduction & Behavioral Fingerprinting

### Overview
The framework implements sophisticated adaptive sampling strategies to handle challenging verification scenarios, particularly when models show high variance or converge to stable intermediate states.

### Key Components

#### 1. Adaptive Sampling (`src/pot/core/adaptive_sampling.py`)
- **AdaptiveSequentialTester**: Wraps base tester with adaptive strategies
- **ConvergenceMetrics**: Tracks mean stability, CI improvement, RME history
- **VarianceReductionStrategy**: Implements stratified sampling, importance sampling, control variates

#### 2. Strategy Switching
The framework automatically switches strategies based on convergence patterns:

```python
# Triggers at 50% budget if stuck in UNDECIDED
if n > n_max * 0.5 and all(d == "UNDECIDED" for d in recent_decisions):
    if abs(mean) < 0.05:  # Near zero mean
        return "symmetric_kl"  # More sensitive metric
    if variance > 0.1:  # High variance
        return "increase_k"  # More positions per prompt
```

**Available Strategies**:
- **`increase_k`**: Increases positions per prompt (default 8→12→16)
- **`symmetric_kl`**: Switches to symmetric KL divergence (more sensitive)
- **`variance_reduction`**: Applies importance sampling and control variates
- **Batch size adaptation**: Dynamically adjusts (8→6→4) near decision boundaries

#### 3. Behavioral Fingerprinting Detection
**Problem**: Models can converge to stable intermediate values that don't meet SAME or DIFFERENT thresholds, causing infinite loops.

**Solution**: Detect and classify stable intermediate states:

```python
# In diff_decision.py around line 285
if self.n >= max(50, self.config.n_min * 2):
    # Check for stable convergence
    cv = self.variance / abs(self.mean) if abs(self.mean) > 1e-10 else float('inf')
    
    if cv < 0.1:  # Low coefficient of variation
        # Classify relationship based on effect size
        if abs(self.mean) < 0.001:
            relationship = "NEAR_CLONE"
        elif abs(self.mean) < 0.01:
            relationship = "SAME_ARCH_FINE_TUNED"
        elif abs(self.mean) < 0.1:
            relationship = "SAME_ARCH_DIFFERENT_SCALE"
        else:
            relationship = "DIFFERENT_ARCH_SIMILAR_TRAINING"
        
        return "UNDECIDED_STABLE", relationship
```

### Real-World Example: Llama-2-7B Testing

Successfully detected fine-tuning differences between base and chat models:

```
# A|B Test (Base vs Chat)
[00:06:23] Strategy switch at n=64: increase_k
[00:20:28] Strategy switch at n=72: increase_k  
[00:35:11] Strategy switch at n=80: increase_k
[00:48:29] Adaptive DIFFERENT decision at n=88: CI [0.033, 4.166]

# Key achievements:
- Detected subtle fine-tuning differences in 7B models
- Prevented infinite loop with behavioral fingerprinting
- Applied 3 adaptive strategies for variance reduction
- Achieved 97.5% confidence on consumer hardware (32GB RAM)
```

### Numerical Stability Fixes

**Critical Fix for Near-Zero Variance**:
```python
# Line 285 in diff_decision.py
epsilon = 1e-10
safe_mean = max(abs(self.mean), epsilon)
rme = half_width / safe_mean  # Prevents division by zero
```

**Convergence Detection Enhancement**:
```python
# In adaptive_sampling.py
if all(d in ["UNDECIDED", "UNDECIDED_STABLE"] for d in recent_decisions[-10:]):
    if any("STABLE" in d for d in recent_decisions[-3:]):
        return True, "Behavioral fingerprint detected"
```

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
huggingface-cli download gpt2 --local-dir ~/LLM_Models/gpt2 --local-dir-use-symlinks False

# For gated models (after huggingface-cli login)
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ~/LLM_Models/llama-2-7b-chat-hf --local-dir-use-symlinks False
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
pwd                          # Should be ~/PoT_Experiments
python --version            # Should be 3.11.8
python -c "import torch, transformers, numpy, scipy; print('✅ Core deps OK')"
ls -la ~/LLM_Models/  # Verify models exist
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pot` | Run from `~/PoT_Experiments` |
| `Permission denied` | `chmod +x scripts/*.sh scripts/*.py` |
| ZK tests fail | Use `--skip-zk` flag or install Rust |
| Out of memory | Use smaller models or `--skip-zk` |
| Models not found | Use open models (gpt2, distilgpt2) |
| CI/CD components warnings | Normal if `tests.fixtures` not installed - core PoT still works |
| **Test runs indefinitely** | Behavioral fingerprinting issue - check fixes below |
| **Division by zero in RME** | Update diff_decision.py line 285 with epsilon guard |
| **High variance, no convergence** | Adaptive sampling will trigger strategy switches |

### Behavioral Fingerprinting Troubleshooting

If tests run indefinitely without reaching a decision:

1. **Check for stable intermediate convergence**:
   ```bash
   # Look for patterns where variance approaches zero but mean is non-zero
   tail -f experimental_results/*/pipeline_results_*.json | grep -E "mean|variance|decision"
   ```

2. **Verify numerical stability fixes are applied**:
   ```bash
   # Check diff_decision.py has epsilon guard
   grep -n "epsilon = 1e-10" src/pot/core/diff_decision.py
   # Should be around line 285
   ```

3. **Monitor adaptive strategy switches**:
   ```bash
   # Watch for strategy switch messages
   grep -E "Strategy switch|increase_k|symmetric_kl" logs/*.log
   ```

4. **Force early termination for debugging**:
   ```python
   # In diff_decision.py, add emergency exit
   if self.n > 100:  # Emergency cap for debugging
       logger.warning(f"Emergency exit at n={self.n}, mean={self.mean:.4f}")
       return "UNDECIDED_STABLE", {"reason": "emergency_cap"}
   ```

### Emergency Reset
```bash
pkill -f python && pkill -f cargo
rm -rf /tmp/pot_* experimental_results/temp_*
cd ~/PoT_Experiments
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
      model_path: ~/LLM_Models/gpt2
    cand:
      type: hf_local
      model_path: ~/LLM_Models/distilgpt2
```


## Remember
This framework validates academic paper claims. **Never create mock tests** - always use the real PoT framework code. Tests should be comprehensive and take several minutes to run, generating real metrics and evidence bundles.
