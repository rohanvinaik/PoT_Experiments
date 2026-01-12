# Behavioral Fingerprinting Results

**Generated:** 2026-01-11 18:44:10

## Summary

| Metric | Value |
|--------|-------|
| Total Experiments | 9 |
| Correct Classifications | 6 |
| **Classification Accuracy** | **66.7%** |
| Avg Queries | 126.1 |
| Avg Time | 9.9s |
| Avg Confidence | 41.82% |

## Results by Category

| Category | Experiments | Correct | Accuracy | Avg Mean Effect | Avg CV |
|----------|-------------|---------|----------|-----------------|--------|
| Architecture | 3 | 2 | 66.7% | 6.3423 | 0.16 |
| Distillation | 1 | 1 | 100.0% | 0.5506 | 0.13 |
| Scale | 3 | 3 | 100.0% | 0.6394 | 0.92 |
| Self Consistency | 2 | 0 | 0.0% | 0.0000 | nan |

## Detailed Results

| Experiment | Category | Expected | Actual | Match | Mean Effect | Variance | CV | Queries |
|------------|----------|----------|--------|-------|-------------|----------|-----|---------|
| self_gpt2 | self_consistency | IDENTICAL | INCONCLUSIVE | NO | 0.0000 | 0.000000 | inf | 30 |
| self_pythia160m | self_consistency | IDENTICAL | INCONCLUSIVE | NO | 0.0000 | 0.000000 | inf | 30 |
| distill_gpt2 | distillation | DISTILLED | DISTILLED | YES | 0.5506 | 0.004861 | 0.13 | 200 |
| scale_pythia_70m_160m | scale | SAME_ARCH_DIFF_SCALE | SAME_ARCHITECTURE_DIFFERENT_SCALE | YES | 0.4922 | 0.097896 | 0.64 | 200 |
| scale_pythia_160m_410m | scale | SAME_ARCH_DIFF_SCALE | SAME_ARCHITECTURE_DIFFERENT_SCALE | YES | 0.5303 | 0.192478 | 0.83 | 200 |
| scale_gpt2_gpt2medium | scale | SAME_ARCH_DIFF_SCALE | SAME_ARCHITECTURE_DIFFERENT_SCALE | YES | 0.8957 | 1.348560 | 1.30 | 200 |
| arch_gpt2_pythia | architecture | DIFFERENT_ARCH | DIFFERENT_ARCHITECTURE | YES | 9.8701 | 0.935228 | 0.10 | 39 |
| arch_gpt2_neo | architecture | DIFFERENT_ARCH | DISTILLED | NO | 0.4782 | 0.020046 | 0.30 | 200 |
| arch_pythia_neo | architecture | DIFFERENT_ARCH | DIFFERENT_ARCHITECTURE | YES | 8.6788 | 0.401432 | 0.07 | 36 |

## Inference Explanations

### self_gpt2 (INCORRECT)
- **Expected:** IDENTICAL
- **Actual:** INCONCLUSIVE
- **Traditional Decision:** IDENTICAL
- **Confidence:** 10.50%
- **Explanation:** 

### self_pythia160m (INCORRECT)
- **Expected:** IDENTICAL
- **Actual:** INCONCLUSIVE
- **Traditional Decision:** IDENTICAL
- **Confidence:** 10.50%
- **Explanation:** 

### distill_gpt2 (CORRECT)
- **Expected:** DISTILLED
- **Actual:** DISTILLED
- **Traditional Decision:** DIFFERENT
- **Confidence:** 73.22%
- **Explanation:** Large stable difference (mean=0.551, CV=0.13) characteristic of student-teacher distillation

### scale_pythia_70m_160m (CORRECT)
- **Expected:** SAME_ARCH_DIFF_SCALE
- **Actual:** SAME_ARCHITECTURE_DIFFERENT_SCALE
- **Traditional Decision:** DIFFERENT
- **Confidence:** 60.61%
- **Explanation:** Moderate mean effect (0.492) with variance ratio 97896.4x suggests same architecture at different scales (e.g., 125M vs 1.3B parameters)

### scale_pythia_160m_410m (CORRECT)
- **Expected:** SAME_ARCH_DIFF_SCALE
- **Actual:** SAME_ARCHITECTURE_DIFFERENT_SCALE
- **Traditional Decision:** DIFFERENT
- **Confidence:** 44.82%
- **Explanation:** Moderate mean effect (0.530) with variance ratio 192478.2x suggests same architecture at different scales (e.g., 125M vs 1.3B parameters)

### scale_gpt2_gpt2medium (CORRECT)
- **Expected:** SAME_ARCH_DIFF_SCALE
- **Actual:** SAME_ARCHITECTURE_DIFFERENT_SCALE
- **Traditional Decision:** DIFFERENT
- **Confidence:** 41.60%
- **Explanation:** Moderate mean effect (0.896) with variance ratio 1348559.9x suggests same architecture at different scales (e.g., 125M vs 1.3B parameters)

### arch_gpt2_pythia (CORRECT)
- **Expected:** DIFFERENT_ARCH
- **Actual:** DIFFERENT_ARCHITECTURE
- **Traditional Decision:** DIFFERENT
- **Confidence:** 35.15%
- **Explanation:** 

### arch_gpt2_neo (INCORRECT)
- **Expected:** DIFFERENT_ARCH
- **Actual:** DISTILLED
- **Traditional Decision:** DIFFERENT
- **Confidence:** 67.45%
- **Explanation:** Large stable difference (mean=0.478, CV=0.30) characteristic of student-teacher distillation

### arch_pythia_neo (CORRECT)
- **Expected:** DIFFERENT_ARCH
- **Actual:** DIFFERENT_ARCHITECTURE
- **Traditional Decision:** DIFFERENT
- **Confidence:** 32.49%
- **Explanation:** 

## Variance Signature Reference

Based on Section 7 of the PoT paper, behavioral fingerprinting uses variance signatures to classify model relationships:

| Relationship | Mean Effect Range | Variance Pattern | CV Range |
|--------------|-------------------|------------------|----------|
| IDENTICAL | < 1e-6 | Minimal | N/A |
| NEAR_CLONE | < 0.001 | Low | Low |
| SAME_ARCH_DIFF_SCALE | 0.001 - 0.5 | Moderate | < 2.0 |
| SAME_ARCH_FINE_TUNED | 0.01 - 0.1 | Low | Low |
| DISTILLED | > 0.5 | Low | < 1.0 |
| DIFFERENT_ARCH | > 0.1 | High | > 2.0 |