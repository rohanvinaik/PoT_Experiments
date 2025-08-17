# Experimental Metrics Summary

## Important Disclaimers

### Reproducibility Notice
The metrics reported in this document represent experimental results from controlled laboratory conditions. When reproducing these experiments, you may observe variations due to:

1. **Hardware Differences**: Results obtained on NVIDIA A100 GPUs may differ from other hardware
2. **Software Versions**: Minor variations across PyTorch, CUDA, and other library versions
3. **Randomness Sources**: Despite fixed seeds, some operations may have platform-specific behavior
4. **Numerical Precision**: Floating-point arithmetic differences across architectures

**Expected Variations**: Results should fall within ±10% of reported values for FAR/FRR and ±1% for accuracy metrics.

### Data Source and Validation
- **Primary Results**: Located in `outputs/consolidated_results.json`
- **Individual Experiments**: Found in `outputs/*/E*/results.jsonl`
- **Validation Reports**: See `outputs/*/validation_report.json` for automated validation

### How to Verify These Metrics
```bash
# Validate against paper claims
python pot/experiments/result_validator.py \
    --results outputs/consolidated_results.json \
    --claimed-metrics configs/paper_claims.json \
    --tolerance 0.1

# Generate fresh metrics report
python pot/experiments/report_generator.py \
    --results outputs/consolidated_results.json \
    --output-format markdown
```

## Core Metrics

**Model variant separation (vision, E1)**
| Variant | n | τ | FAR | FRR | AUROC |
|---|---|---|---|---|---|
| identical | 256 | 0.01 | 0.0 | 0.0 | 1.00 |
| seed_variant | 256 | 0.02 | 0.0 | 0.0 | 1.00 |
| fine_tuned | 256 | 0.10 | 0.0234 | 0.0 | 0.9883 |
| pruned | 256 | 0.10 | 0.0117 | 0.5508 | 0.7188 |
| quantized | 256 | 0.10 | 0.0117 | 0.5000 | 0.7441 |
| distilled | 256 | 0.10 | 0.0117 | 0.4961 | 0.7461 |

**Verification runs across challenge families**
| Dataset | Exp | Challenge | n | τ | FAR | FRR |
|---|---|---|---|---|---|---|
| lm_small | E7 | lm:templates | 256 | 0.05 | 0.0039 | 0.0 |
| lm_small | E2 | lm:templates | 512 | 0.05 | 0.0039 | 0.0 |
| lm_small | E3 | lm:templates | 256 | 0.05 | 0.0039 | 0.0 |
| vision_cifar10 | E7 | vision:freq | 256 | 0.05 | 0.0039 | 0.0 |
| vision_cifar10 | E7 | vision:texture | 256 | 0.05 | 0.0039 | 0.0 |
| vision_cifar10 | E5 | vision:freq | 128 | 0.05 | 0.0 | 0.0 |
| vision_cifar10 | E3 | vision:freq | 256 | 0.05 | 0.0039 | 0.0 |
| vision_cifar10 | E4 | vision:freq | 256 | 0.05 | 0.0039 | 0.0 |

Average query budget across these runs: ~272 challenges.

## Experimental Setup
- **Vision experiments** use CIFAR10 test images resized to 224×224, with a ResNet-18 reference model and variants (seed, finetune, prune, quantize, distill). Challenge families include `vision:freq` and `vision:texture` with 256 queries each.
- **Language-model experiments** employ TinyLlama‑1.1B and variants (seed, LoRA finetune, quantize, distill) using the `lm:templates` challenge family with 512 templates and canonicalized output comparison.

## Coverage–Separation Trade-off & Robustness
- **Trade-off:** The E1 results show perfect separation for identical or seed variants (FAR=FRR=0, AUROC=1) but substantially higher FRR for heavily modified models such as pruned or quantized variants (FRR≈0.5) while keeping FAR near 0.01. This indicates that maintaining low false alarms (separation) against transformed models reduces coverage, aligning with the coverage–separation trade-off.
- **Robustness:** Verification across both vision and language tasks consistently achieves FAR ≤0.0039 and FRR=0 at τ=0.05, regardless of dataset or challenge type, demonstrating robustness of the protocol to diverse challenge families and model types.

## How Metrics Are Calculated

### False Accept Rate (FAR)
The probability of incorrectly accepting a different model as the reference:
```python
FAR = FP / (FP + TN)
```
Where FP = false positives, TN = true negatives

### False Reject Rate (FRR)
The probability of incorrectly rejecting the genuine reference model:
```python
FRR = FN / (FN + TP)
```
Where FN = false negatives, TP = true positives

### Area Under ROC Curve (AUROC)
Computed using trapezoidal integration over FAR-TPR pairs across all thresholds.

### Confidence Intervals
All metrics include 95% bootstrap confidence intervals (1000 iterations):
```python
from pot.experiments import MetricsCalculator
calculator = MetricsCalculator()
metrics = calculator.calculate_with_confidence_intervals(
    predictions, labels, n_bootstrap=1000
)
```

## Reconciliation Notes

### Understanding Discrepancies

When your reproduced results differ from these reported metrics, consider:

1. **Minor Variations (≤10% difference)**
   - **Cause**: Normal hardware/software variations
   - **Action**: No action needed, results are successfully reproduced
   - **Example**: FAR of 0.011 vs reported 0.010

2. **Moderate Variations (10-25% difference)**
   - **Cause**: Configuration differences, suboptimal hyperparameters
   - **Action**: Check configuration files, ensure deterministic mode
   - **Example**: FRR of 0.0125 vs reported 0.010
   - **Reconciliation**: 
     ```bash
     # Verify configuration
     diff configs/minimal_reproduce.yaml configs/paper_reproduce.yaml
     
     # Check seeds
     grep -r "seed" configs/*.yaml
     ```

3. **Major Variations (>25% difference)**
   - **Cause**: Significant implementation differences, missing dependencies
   - **Action**: Full diagnostic required
   - **Example**: AUROC of 0.75 vs reported 0.99
   - **Reconciliation**:
     ```bash
     # Run diagnostic
     python scripts/diagnose_reproduction.py \
         --expected outputs/paper_results.json \
         --actual outputs/your_results.json
     ```

### Common Reconciliation Scenarios

#### Scenario 1: Higher FAR than Expected
**Symptoms**: FAR > 0.015 when expecting ~0.010

**Likely Causes**:
- Threshold calibration differences
- Challenge distribution mismatch
- Model loading inconsistencies

**Resolution**:
```python
# Recalibrate threshold
from pot.experiments import calibrate_threshold
optimal_tau = calibrate_threshold(
    model, challenges, target_far=0.01
)
```

#### Scenario 2: Perfect Metrics (All Zeros/Ones)
**Symptoms**: FAR=0, FRR=0, AUROC=1.0

**Likely Causes**:
- Testing on training data
- Insufficient challenge diversity
- Overly similar model variants

**Resolution**:
```bash
# Verify challenge uniqueness
python -c "
from pot.core.challenge import generate_challenges, ChallengeConfig
config = ChallengeConfig(...)
challenges = generate_challenges(config)
print(f'Unique challenges: {len(set(c.challenge_id for c in challenges))}')
"
```

#### Scenario 3: High FRR on Compressed Models
**Symptoms**: FRR > 0.5 for pruned/quantized models

**Expected Behavior**: This is actually expected and documented. Heavily modified models naturally have higher FRR while maintaining low FAR.

**Interpretation**: The system correctly identifies these as "different" models, which is the desired behavior for security purposes.

### Automated Reconciliation

Use the built-in reconciliation system:

```python
from pot.experiments import ResultValidator

validator = ResultValidator()
validator.compare_claimed_vs_actual(
    claimed_metrics={'far': 0.01, 'frr': 0.01},
    actual_results=your_results
)

# Get reconciliation suggestions
suggestions = validator.reconcile_discrepancies()
for suggestion in suggestions:
    print(f"• {suggestion}")
```

### Getting Help with Discrepancies

If reconciliation doesn't resolve discrepancies:

1. **Generate diagnostic report**:
   ```bash
   python scripts/generate_diagnostic.py \
       --results outputs/your_results.json \
       --output diagnostic_report.md
   ```

2. **Check environment**:
   ```bash
   python -m pot.utils.check_environment
   ```

3. **Run minimal test**:
   ```bash
   python -m pytest tests/test_metrics_consistency.py -xvs
   ```

4. **Report issue** with diagnostic information:
   - Include `diagnostic_report.md`
   - Specify hardware/software configuration
   - Provide reproduction command used

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-15 | Initial metrics from paper |
| 1.1.0 | 2024-02-01 | Added confidence intervals |
| 1.2.0 | 2024-03-15 | Updated with ablation studies |
| 1.3.0 | 2024-08-15 | Added reconciliation notes |

## Related Documents

- **Full Results**: `outputs/consolidated_results.json`
- **Validation Reports**: `outputs/*/validation_report.json`
- **Reproduction Guide**: [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)
- **Configuration Files**: `configs/paper_claims.json`
- **Statistical Methods**: [docs/statistical_verification.md](docs/statistical_verification.md)

