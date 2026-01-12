# Behavioral Fingerprinting Results

**Generated:** 2026-01-11 18:26:36

## Summary

| Metric | Value |
|--------|-------|
| Total Experiments | 9 |
| Correct Classifications | 0 |
| **Accuracy** | **0.0%** |
| Avg Queries | 190.6 |
| Avg Time | 14.8s |

## Results by Category

| Category | Experiments | Correct | Accuracy |
|----------|-------------|---------|----------|
| Architecture | 3 | 0 | 0.0% |
| Distillation | 1 | 0 | 0.0% |
| Scale | 3 | 0 | 0.0% |
| Self Consistency | 2 | 0 | 0.0% |

## Detailed Results

| Experiment | Category | Ref | Cand | Expected | Actual | Correct | Mean Effect | Variance | CV |
|------------|----------|-----|------|----------|--------|---------|-------------|----------|-----|
| self_gpt2 | self_consistency | gpt2 | gpt2 | IDENTICAL | DIFFERENT_ARCHITECTURE | ✗ | 0.1672 | 0.305179 | 3.30 |
| self_pythia160m | self_consistency | pythia-160m | pythia-160m | IDENTICAL | INCONCLUSIVE | ✗ | nan | nan | inf |
| distill_gpt2 | distillation | gpt2 | distilgpt2 | DISTILLED | DIFFERENT_ARCHITECTURE | ✗ | 1.3715 | 0.337687 | 0.42 |
| scale_pythia_70m_160m | scale | pythia-70m | pythia-160m | SAME_ARCH_DIFF_SCALE | INCONCLUSIVE | ✗ | nan | nan | inf |
| scale_pythia_160m_410m | scale | pythia-160m | pythia-410m | SAME_ARCH_DIFF_SCALE | INCONCLUSIVE | ✗ | nan | nan | inf |
| scale_gpt2_gpt2medium | scale | gpt2 | gpt2-medium | SAME_ARCH_DIFF_SCALE | DIFFERENT_ARCHITECTURE | ✗ | 0.9419 | 1.147098 | 1.14 |
| arch_gpt2_pythia | architecture | gpt2 | pythia-160m | DIFFERENT_ARCH | INCONCLUSIVE | ✗ | nan | nan | inf |
| arch_gpt2_neo | architecture | gpt2 | gpt-neo-125m | DIFFERENT_ARCH | DISTILLED | ✗ | 6.5208 | 2.525329 | 0.24 |
| arch_pythia_neo | architecture | pythia-160m | gpt-neo-125m | DIFFERENT_ARCH | INCONCLUSIVE | ✗ | nan | nan | inf |

## Inference Explanations

### self_gpt2
- **Expected:** IDENTICAL
- **Actual:** DIFFERENT_ARCHITECTURE
- **Explanation:** High variance (ratio=305179.0x) and instability (CV=3.30) indicates fundamentally different architectures
- **Confidence:** 35.00%

### self_pythia160m
- **Expected:** IDENTICAL
- **Actual:** INCONCLUSIVE
- **Explanation:** Statistical patterns do not match known relationships (mean=nan, variance=nan, n=200)
- **Confidence:** 35.00%

### distill_gpt2
- **Expected:** DISTILLED
- **Actual:** DIFFERENT_ARCHITECTURE
- **Explanation:** High variance (ratio=337686.6x) and instability (CV=0.42) indicates fundamentally different architectures
- **Confidence:** 75.25%

### scale_pythia_70m_160m
- **Expected:** SAME_ARCH_DIFF_SCALE
- **Actual:** INCONCLUSIVE
- **Explanation:** Statistical patterns do not match known relationships (mean=nan, variance=nan, n=200)
- **Confidence:** 35.00%

### scale_pythia_160m_410m
- **Expected:** SAME_ARCH_DIFF_SCALE
- **Actual:** INCONCLUSIVE
- **Explanation:** Statistical patterns do not match known relationships (mean=nan, variance=nan, n=200)
- **Confidence:** 35.00%

### scale_gpt2_gpt2medium
- **Expected:** SAME_ARCH_DIFF_SCALE
- **Actual:** DIFFERENT_ARCHITECTURE
- **Explanation:** High variance (ratio=1147098.0x) and instability (CV=1.14) indicates fundamentally different architectures
- **Confidence:** 44.60%

### arch_gpt2_pythia
- **Expected:** DIFFERENT_ARCH
- **Actual:** INCONCLUSIVE
- **Explanation:** Statistical patterns do not match known relationships (mean=nan, variance=nan, n=200)
- **Confidence:** 35.00%

### arch_gpt2_neo
- **Expected:** DIFFERENT_ARCH
- **Actual:** DISTILLED
- **Explanation:** 
- **Confidence:** 81.02%

### arch_pythia_neo
- **Expected:** DIFFERENT_ARCH
- **Actual:** INCONCLUSIVE
- **Explanation:** Statistical patterns do not match known relationships (mean=nan, variance=nan, n=200)
- **Confidence:** 35.00%

## Variance Signature Analysis

The behavioral fingerprinting uses variance signatures to discriminate between model relationships:

| Relationship | Typical Mean Effect | Typical Variance | Typical CV |
|--------------|---------------------|------------------|------------|
| IDENTICAL | < 1e-6 | < 1e-5 | - |
| SAME_ARCH_DIFF_SCALE | 0.001 - 0.5 | Low-Moderate | < 2.0 |
| DISTILLED | > 0.5 | Low | < 1.0 |
| DIFFERENT_ARCH | > 0.1 | High | > 2.0 |