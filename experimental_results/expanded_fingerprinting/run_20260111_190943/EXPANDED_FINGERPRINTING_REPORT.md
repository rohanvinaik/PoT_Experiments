# Expanded Behavioral Fingerprinting Results

**Generated:** 2026-01-11 19:11:39

## Summary

| Metric | Value |
|--------|-------|
| Total Experiments | 12 |
| Correct Classifications | 10 |
| **Classification Accuracy** | **83.3%** |
| Avg Queries | 129.2 |
| Avg Time | 7.7s |

## Results by Category

| Category | Experiments | Correct | Accuracy |
|----------|-------------|---------|----------|
| Architecture | 3 | 2 | 66.7% |
| Cross Scale | 1 | 1 | 100.0% |
| Deduplication | 1 | 0 | 0.0% |
| Distillation | 1 | 1 | 100.0% |
| Finetune Dialog | 1 | 1 | 100.0% |
| Scale | 3 | 3 | 100.0% |
| Self Consistency | 2 | 2 | 100.0% |

## Detailed Results

| Experiment | Category | Expected | Actual | Match | Mean Effect | CV | Confidence |
|------------|----------|----------|--------|-------|-------------|-----|------------|
| self_gpt2 | self_consistency | IDENTICAL | IDENTICAL | ✓ | 0.0000 | ∞ | 95% |
| self_pythia160m | self_consistency | IDENTICAL | IDENTICAL | ✓ | 0.0000 | ∞ | 95% |
| distill_gpt2 | distillation | DISTILLED | DISTILLED | ✓ | 0.6051 | 0.18 | 85% |
| scale_pythia_70m_160m | scale | SAME_ARCH_DIFF_SCALE | SAME_ARCHITECTURE_DIFFERENT_SCALE | ✓ | 0.5841 | 0.40 | 75% |
| scale_pythia_160m_410m | scale | SAME_ARCH_DIFF_SCALE | SAME_ARCHITECTURE_DIFFERENT_SCALE | ✓ | 0.6182 | 0.71 | 75% |
| scale_gpt2_medium | scale | SAME_ARCH_DIFF_SCALE | SAME_ARCHITECTURE_DIFFERENT_SCALE | ✓ | 0.7918 | 1.22 | 75% |
| finetune_dialog_small | finetune_dialog | FINE_TUNED | EXTENSIVE_FINE_TUNING | ✓ | 2.8775 | 0.39 | 75% |
| dedup_pythia_160m | deduplication | DEDUPLICATED | SAME_ARCHITECTURE_DIFFERENT_SCALE | ✗ | 0.3579 | 0.45 | 75% |
| arch_gpt2_pythia | architecture | DIFFERENT_ARCH | DIFFERENT_ARCHITECTURE | ✓ | 9.3094 | 0.10 | 95% |
| arch_gpt2_neo | architecture | DIFFERENT_ARCH | FINE_TUNED | ✗ | 0.5323 | 0.26 | 70% |
| arch_pythia_neo | architecture | DIFFERENT_ARCH | DIFFERENT_ARCHITECTURE | ✓ | 8.3599 | 0.10 | 95% |
| cross_gpt2_pythia70m | cross_scale | DIFFERENT_ARCH | DIFFERENT_ARCHITECTURE | ✓ | 8.7539 | 0.10 | 95% |

## Inference Explanations

### self_gpt2 (CORRECT)
- **Description:** GPT-2 vs itself - baseline self-consistency
- **Expected:** IDENTICAL
- **Actual:** IDENTICAL
- **Confidence:** 95.0%
- **Explanation:** Zero divergence - identical model weights

### self_pythia160m (CORRECT)
- **Description:** Pythia-160M vs itself - baseline self-consistency
- **Expected:** IDENTICAL
- **Actual:** IDENTICAL
- **Confidence:** 95.0%
- **Explanation:** Zero divergence - identical model weights

### distill_gpt2 (CORRECT)
- **Description:** GPT-2 vs DistilGPT-2 - knowledge distillation
- **Expected:** DISTILLED
- **Actual:** DISTILLED
- **Confidence:** 85.0%
- **Explanation:** Moderate stable difference (mean=0.605, CV=0.18) - characteristic of distillation

### scale_pythia_70m_160m (CORRECT)
- **Description:** Pythia 70M vs 160M - same architecture, different scale
- **Expected:** SAME_ARCH_DIFF_SCALE
- **Actual:** SAME_ARCHITECTURE_DIFFERENT_SCALE
- **Confidence:** 75.0%
- **Explanation:** Moderate variable difference (mean=0.584, CV=0.40) - scale variation

### scale_pythia_160m_410m (CORRECT)
- **Description:** Pythia 160M vs 410M - same architecture, different scale
- **Expected:** SAME_ARCH_DIFF_SCALE
- **Actual:** SAME_ARCHITECTURE_DIFFERENT_SCALE
- **Confidence:** 75.0%
- **Explanation:** Moderate variable difference (mean=0.618, CV=0.71) - scale variation

### scale_gpt2_medium (CORRECT)
- **Description:** GPT-2 (124M) vs GPT-2 Medium (355M) - same architecture, different scale
- **Expected:** SAME_ARCH_DIFF_SCALE
- **Actual:** SAME_ARCHITECTURE_DIFFERENT_SCALE
- **Confidence:** 75.0%
- **Explanation:** Moderate variable difference (mean=0.792, CV=1.22) - scale variation

### finetune_dialog_small (CORRECT)
- **Description:** GPT-2 vs DialoGPT-small - fine-tuned for conversation
- **Expected:** FINE_TUNED
- **Actual:** EXTENSIVE_FINE_TUNING
- **Confidence:** 75.0%
- **Explanation:** Extensive divergence (mean=2.88) - heavily modified or different architecture

### dedup_pythia_160m (INCORRECT)
- **Description:** Pythia-160M vs Pythia-160M-deduped - deduplicated training data
- **Expected:** DEDUPLICATED
- **Actual:** SAME_ARCHITECTURE_DIFFERENT_SCALE
- **Confidence:** 75.0%
- **Explanation:** Moderate variable difference (mean=0.358, CV=0.45) - scale variation

### arch_gpt2_pythia (CORRECT)
- **Description:** GPT-2 vs Pythia-160M - different architectures
- **Expected:** DIFFERENT_ARCH
- **Actual:** DIFFERENT_ARCHITECTURE
- **Confidence:** 95.0%
- **Explanation:** Large divergence (mean=9.31) - clearly different architecture

### arch_gpt2_neo (INCORRECT)
- **Description:** GPT-2 vs GPT-Neo-125M - different architectures
- **Expected:** DIFFERENT_ARCH
- **Actual:** FINE_TUNED
- **Confidence:** 70.0%
- **Explanation:** Moderate difference (mean=0.532, CV=0.26) - likely fine-tuning

### arch_pythia_neo (CORRECT)
- **Description:** Pythia-160M vs GPT-Neo-125M - different architectures
- **Expected:** DIFFERENT_ARCH
- **Actual:** DIFFERENT_ARCHITECTURE
- **Confidence:** 95.0%
- **Explanation:** Large divergence (mean=8.36) - clearly different architecture

### cross_gpt2_pythia70m (CORRECT)
- **Description:** GPT-2 (124M) vs Pythia-70M - different arch, similar scale
- **Expected:** DIFFERENT_ARCH
- **Actual:** DIFFERENT_ARCHITECTURE
- **Confidence:** 95.0%
- **Explanation:** Large divergence (mean=8.75) - clearly different architecture

