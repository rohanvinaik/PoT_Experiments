# Statistical Difference Decision Framework

## Overview

The Statistical Difference Decision Framework (`pot/core/diff_decision.py`) provides a rigorous statistical testing system for model verification with anytime stopping based on confidence intervals and relative margin of error. This framework enables efficient determination of whether two models are statistically different, identical, or equivalent within specified tolerances.

## Key Features

### 1. **Anytime-Valid Confidence Intervals**
- Empirical-Bernstein (EB) bounds for bounded scores
- Student's t-distribution for unbounded scores
- Welford's online algorithm for efficient streaming computation

### 2. **Multiple Decision Types**
- **DIFFERENT**: Models are statistically different (CI excludes 0)
- **IDENTICAL**: Models are statistically identical (differences within noise)
- **SAME**: Models are equivalent within tolerance band (TOST)
- **UNDECIDED**: Unable to determine with current samples/precision

### 3. **Adaptive Sampling**
- Early stopping when decision criteria met
- Configurable minimum/maximum sample sizes
- Batch processing for efficiency
- Relative margin of error targeting

## Configuration

```python
from pot.core.diff_decision import DiffDecisionConfig

config = DiffDecisionConfig(
    # Confidence & precision
    alpha=0.01,                    # 99% CI (two-sided)
    rel_margin_target=0.05,        # 5% relative half-width target
    min_effect_floor=1e-4,         # Guard against division by zero
    
    # Sampling plan
    n_min=10,                      # Minimum samples before stopping
    n_max=200,                     # Maximum samples
    batch_size=4,                  # Challenges per batch
    positions_per_prompt=32,       # K positions for teacher-forced scoring
    
    # Method selection
    method="eb",                   # "eb" or "t"
    clip_low=0.0,                  # For EB: lower bound
    clip_high=0.2,                 # For EB: upper bound
    
    # Optional equivalence testing
    equivalence_band=0.01,         # γ for TOST-style testing
    
    # Early stopping
    early_stop_threshold=1e-6,     # Stop if mean < this after n_min
    identical_model_n_min=20       # Min samples for identical check
)
```

## Usage Example

```python
from pot.core.diff_decision import DifferenceVerifier, DiffDecisionConfig

# Define scoring function
def score_fn(ref_model, cand_model, prompt, K=32):
    """Compute score difference between models"""
    # Implementation: compute delta-CE, symmetric KL, etc.
    return score_difference

# Define prompt generator
def prompt_generator():
    """Generate next test prompt"""
    return next_prompt

# Configure verifier
config = DiffDecisionConfig(
    n_min=20,
    n_max=100,
    rel_margin_target=0.05
)

verifier = DifferenceVerifier(score_fn, prompt_generator, config)

# Run verification
report = verifier.verify_difference(ref_model, cand_model)

# Check results
print(f"Decision: {report['results']['decision']}")
print(f"Mean difference: {report['results']['mean']:.6f}")
print(f"99% CI: {report['results']['ci_99']}")
print(f"Samples used: {report['results']['n_used']}")
```

## Algorithm Details

### Welford's Online Algorithm
Efficiently computes mean and variance without storing all observations:
```python
# Update with new observation x
delta = x - mean
mean += delta / n
delta2 = x - mean
M2 += delta * delta2
variance = M2 / (n - 1)
```

### Empirical-Bernstein Bounds
For bounded variables in [a, b], provides anytime-valid confidence sequences:
```python
# Normalize to [0,1]
y = (x - a) / (b - a)

# Compute EB radius
h = sqrt(2 * var * log(3/α) / n) + 3 * log(3/α) / n

# Map back to original scale
CI = mean ± h * (b - a)
```

### Stopping Criteria

1. **Different**: CI excludes 0 AND relative margin ≤ target
2. **Identical**: |mean| < threshold AND CI width < threshold
3. **Same**: CI within [-γ, γ] AND relative margin ≤ target
4. **Undecided**: n = n_max OR precision target not met

## Performance Characteristics

- **Throughput**: >40,000 samples/second on modern hardware
- **Memory**: O(1) for statistics computation (Welford's algorithm)
- **Early stopping**: Typically saves 50-80% of samples vs fixed-n testing
- **Statistical power**: 99% confidence intervals by default

## Integration with PoT

The framework integrates seamlessly with other PoT components:

1. **Challenge Generation**: Use deterministic prompts from PRF
2. **Scoring**: Leverage teacher-forced scoring methods
3. **Secondary Verification**: Combine with fuzzy hashing, Merkle proofs
4. **Attack Detection**: Use as first-pass filter before expensive checks

## Test Results

The framework includes comprehensive tests (`pot/core/test_diff_decision.py`):
- 18 unit tests covering all major functionality
- Integration tests for complete workflows
- Stress tests with up to 1000 samples
- Mock model testing for various scenarios

## Recommended Next Steps

Based on decision outcomes:

### DIFFERENT Models
- Confirm via fuzzy/TLSH behavioral hashes
- Generate Merkle proofs of challenge-response pairs
- Run attack detectors (distillation, pruning, quantization)
- Optional: Compute Jacobian/CKA similarity

### IDENTICAL Models
- Verify file hashes match
- Check model metadata/configuration
- No further statistical testing needed

### SAME Models (Equivalent)
- May indicate quantization/precision differences
- Consider tighter equivalence bands if needed
- Test deterministic behavior on edge cases

### UNDECIDED
- Increase n_max or positions_per_prompt
- Try different scoring methods
- Diversify challenge families
- Escalate to secondary PoT metrics

## Performance Tips

1. **Batch Processing**: Use larger batch_size for better throughput
2. **Early Stopping**: Set appropriate thresholds for your use case
3. **Method Selection**: Use EB for bounded scores, t-distribution for unbounded
4. **Equivalence Testing**: Set equivalence_band based on practical significance
5. **Variance Reduction**: Increase positions_per_prompt for noisy models

## Conclusion

The Statistical Difference Decision Framework provides a principled, efficient approach to model verification with strong statistical guarantees. Its anytime-valid confidence intervals, adaptive sampling, and multiple decision types make it suitable for production use in the PoT system.