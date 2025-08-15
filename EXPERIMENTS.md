# Proof-of-Training Experiments

## Overview

This document describes the experimental framework for validating Proof-of-Training (PoT) systems. The experiments test separation, query budgets, leakage resistance, non-IID drift, adversarial attacks, sequential testing, and baseline comparisons.

## Experiment Set

### E1: Separation vs. Query Budget (Core Claim)

**Purpose**: Validate that behavioral fingerprints provide strong separation between models with reasonable query budgets.

**Setup**:
- Model pairs: (reference vs identical), (reference vs seed-variant), (reference vs fine-tune), (reference vs quantized), (reference vs pruned), (reference vs distilled)
- Challenge sizes: n ∈ {32, 64, 128, 256, 512, 1024}
- Metrics: FAR/FRR ROC curves, DET curves, AUROC vs n

**Expected Results**:
- Identical models: Near-perfect separation (AUROC > 0.99)
- Seed variants: Strong separation (AUROC > 0.95) with n ≥ 256
- Fine-tuned/modified: Good separation (AUROC > 0.90) with n ≥ 512

**Run**:
```bash
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type roc
```

### E2: Leakage Ablation (Theorem 2 Empirical)

**Purpose**: Test robustness to challenge leakage.

**Setup**:
- Leakage fractions: ρ ∈ {0, 0.1, 0.25, 0.5}
- Attack: Targeted fine-tuning on leaked challenges
- Metric: Detection rate vs ρ

**Expected Results**:
- Detection rate scales approximately as (1-ρ)
- Even with 25% leakage, maintain > 70% detection rate

**Run**:
```bash
python scripts/run_attack.py --config configs/lm_small.yaml --attack targeted_finetune --rho 0.25
python scripts/run_verify.py --config configs/lm_small.yaml --challenge_family lm:templates --n 512
python scripts/run_plots.py --exp_dir outputs/lm_small/E2 --plot_type leakage
```

### E3: Non-IID Drift & Determinism Stress

**Purpose**: Test robustness to distribution shifts and hardware variations.

**Setup**:
- Vision: Preprocessing perturbations (interpolation changes, resize jitter)
- LM: Varied max length, GPU/CPU/mixed precision
- Metric: FAR/FRR stability under drift

**Expected Results**:
- FAR/FRR degradation < 10% under minor drift
- Maintain verification accuracy across hardware

### E4: Adversarial Attacks

**Purpose**: Evaluate resistance to active attacks.

**Attack Types**:
1. **Wrapper Mapping**: Temperature scaling + bias fitting
2. **Targeted Fine-tune**: Training on leaked challenges
3. **Limited Distillation**: Student model with query budget

**Metrics**:
- (Attack cost, queries) → Detection rate trade-off
- Success rate vs attack budget

**Run**:
```bash
python scripts/run_attack.py --config configs/vision_cifar10.yaml --attack wrapper
python scripts/run_attack.py --config configs/vision_cifar10.yaml --attack distillation --budget 10000
```

### E5: Sequential Testing

**Purpose**: Reduce query requirements through early stopping.

**Setup**:
- SPRT/e-values implementation
- Target FAR/FRR levels
- Metric: Average queries to decision

**Expected Results**:
- 30-50% query reduction with sequential testing
- Maintain target error rates

### E6: Baseline Comparisons

**Purpose**: Compare against simpler verification methods.

**Baselines**:
1. Naive I/O hash of raw outputs
2. Lightweight fingerprinting

**Metrics**:
- Separation quality (AUROC)
- Query requirements
- Robustness to variations

### E7: Ablation Studies

**Purpose**: Understand contribution of individual components.

**Ablations**:
- Quantization precision: p ∈ {3, 4, 6}
- Distance metrics: L2 vs KL (vision), edit vs embed (LM)
- Probe families: freq vs texture (vision), arithmetic vs robustness (LM)

## Experimental Protocol

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set deterministic mode
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
```

### 2. Data Preparation

- Vision: CIFAR-10 test set, ImageNet validation subset
- LM: Synthetic prompts (arithmetic, logic, translation)
- Ensure fixed data ordering with seed control

### 3. Model Preparation

**Reference Models**:
- Vision: ResNet18/50 pretrained or trained from scratch
- LM: TinyLlama-1.1B or similar small model

**Variant Generation**:
- Seed variant: Different initialization
- Fine-tuned: 1-2 epochs on subset
- Pruned: 30-40% weight removal
- Quantized: INT8 quantization
- Distilled: Limited query budget training

### 4. Challenge Generation

```python
from pot.core.challenge import ChallengeConfig, generate_challenges

config = ChallengeConfig(
    master_key_hex="0" * 64,
    session_nonce_hex=new_session_nonce(),
    n=256,
    family="vision:freq",
    params={"freq_range": [0.5, 8.0], "contrast_range": [0.3, 1.0]}
)

challenges = generate_challenges(config)
```

### 5. Verification Flow

1. Generate challenges using KDF-seeded PRNG
2. Apply challenges to models
3. Canonicalize outputs (quantization, text normalization)
4. Compute distances
5. Calculate T-statistic and confidence bounds
6. Determine verification decision

### 6. Logging & Artifacts

Each run produces:
- `verify.jsonl`: Per-challenge results
- `summary.json`: Aggregate statistics
- `config_snapshot.yaml`: Full configuration
- Commit hashes and salts for audit

## Acceptance Criteria

### Required Plots
- [x] ROC curves for E1
- [x] DET curves for E1  
- [x] AUROC vs query budget
- [x] Leakage curve for E2
- [x] Robustness curves for E3
- [x] Attack effectiveness for E4
- [x] Query-to-decision for E5

### Required Tables
- Mean ± CI of FAR/FRR at target τ across seeds
- Average queries under sequential testing
- Attack cost vs detection rate

### Required Artifacts
- Challenge IDs and salts
- Commit messages for governance
- Config snapshots per run
- Reproducible seed management

## Implementation Notes

### Determinism Requirements
- `torch.use_deterministic_algorithms(True)`
- Fixed seeds for all RNGs
- Disable dropout/BatchNorm during eval
- LMs: `do_sample=False, temperature=0.0, top_k=1`

### Distance Computation
- Ensure bounded distances for theoretical guarantees
- Report empirical variance
- Use empirical-Bernstein confidence intervals

### Security Considerations
- Call behavioral artifact a "fingerprint" not "signature"
- Avoid prompts that elicit memorized training data
- Use synthetic challenges for privacy

## Quick Start

### Run All Core Experiments
```bash
# E1: Core separation
bash experiments/run_e1.sh

# E2: Leakage
bash experiments/run_e2.sh

# E3: Drift
bash experiments/run_e3.sh

# E4: Attacks
bash experiments/run_e4.sh

# E5: Sequential
bash experiments/run_e5.sh

# Generate all plots
python scripts/generate_all_plots.py
```

### Single Experiment
```bash
# Generate reference
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml

# Run verification
python scripts/run_verify.py --config configs/vision_cifar10.yaml \
    --challenge_family vision:freq --n 256

# Plot results
python scripts/run_plots.py --exp_dir outputs/vision_cifar10 --plot_type roc
```

## Troubleshooting

### Common Issues

1. **Non-deterministic results**: 
   - Ensure `PYTHONHASHSEED=0`
   - Check CUDA deterministic mode
   - Verify fixed dataloader ordering

2. **Memory issues with large models**:
   - Reduce batch size
   - Use gradient checkpointing
   - Run on subset of challenges

3. **Slow verification**:
   - Use smaller challenge sets for development
   - Enable GPU acceleration
   - Consider parallel processing

## References

- Original PoT paper and theoretical foundations
- Empirical-Bernstein concentration bounds
- Sequential testing literature (SPRT, e-values)
- Model fingerprinting related work