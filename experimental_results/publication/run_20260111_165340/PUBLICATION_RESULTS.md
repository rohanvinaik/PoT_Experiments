# Proof-of-Training Verification Results

**Generated:** 2026-01-11T16:53:40.625356
**Test Suite:** PoT Publication Experiment Suite
**Testing Mode:** audit (99% confidence)

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 0.0% |
| **Correct Decisions** | 0 / 12 |
| **Avg Queries/Decision** | 0.0 |
| **Avg Duration** | 0.0s |
| **Errors** | 12 |
| **Skipped** | 0 |

## Results by Category

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| Self Consistency | 0.0% | 0 | 3 |
| Distillation | 0.0% | 0 | 1 |
| Finetuning | 0.0% | 0 | 2 |
| Scale | 0.0% | 0 | 3 |
| Architecture | 0.0% | 0 | 3 |

## Detailed Results

| Experiment | Ref Model | Cand Model | Expected | Actual | Correct | Confidence | Queries | Time |
|------------|-----------|------------|----------|--------|---------|------------|---------|------|
| self_gpt2 | gpt2 | gpt2 | SAME | ERROR | ❌ | 0.000 | 0 | 0.0s |
| self_pythia160m | pythia-160m | pythia-160m | SAME | ERROR | ❌ | 0.000 | 0 | 0.0s |
| self_phi2 | phi-2 | phi-2 | SAME | ERROR | ❌ | 0.000 | 0 | 0.0s |
| distill_gpt2 | gpt2 | distilgpt2 | DIFFERENT | ERROR | ❌ | 0.000 | 0 | 0.0s |
| finetune_llama2_7b | Llama-2-7b-hf | Llama-2-7b-chat-hf | DIFFERENT | ERROR | ❌ | 0.000 | 0 | 0.0s |
| finetune_mistral_zephyr | Mistral-7B-Instruct- | zephyr-7b-beta | DIFFERENT | ERROR | ❌ | 0.000 | 0 | 0.0s |
| scale_pythia_70m_160m | pythia-70m | pythia-160m | DIFFERENT | ERROR | ❌ | 0.000 | 0 | 0.0s |
| scale_pythia_160m_410m | pythia-160m | pythia-410m | DIFFERENT | ERROR | ❌ | 0.000 | 0 | 0.0s |
| scale_gpt_neo | gpt-neo-125m | gpt-neo-1.3B | DIFFERENT | ERROR | ❌ | 0.000 | 0 | 0.0s |
| arch_gpt2_pythia | gpt2 | pythia-160m | DIFFERENT | ERROR | ❌ | 0.000 | 0 | 0.0s |
| arch_tinyllama_phi2 | TinyLlama-1.1B-Chat- | phi-2 | DIFFERENT | ERROR | ❌ | 0.000 | 0 | 0.0s |
| arch_neo_tinyllama | gpt-neo-1.3B | TinyLlama-1.1B-Chat- | DIFFERENT | ERROR | ❌ | 0.000 | 0 | 0.0s |

## Methodology

The Proof-of-Training (PoT) framework uses sequential statistical testing with
Empirical-Bernstein confidence bounds to make anytime-valid decisions about model
identity. The key properties are:

1. **Pre-committed challenges**: HMAC-SHA256 derived challenge seeds prevent cherry-picking
2. **Sequential testing**: Early stopping when sufficient confidence is reached
3. **Separate decision rules**: Distinct criteria for SAME vs DIFFERENT decisions
4. **Behavioral fingerprinting**: Detection of stable intermediate states

### Testing Modes
- **QUICK_GATE**: 97.5% confidence, max 120 queries (screening)
- **AUDIT_GRADE**: 99% confidence, max 400 queries (publication quality)
- **EXTENDED**: 99.9% confidence, max 800 queries (high-stakes)

This experiment suite used **AUDIT_GRADE** mode for publication-quality results.

## Citation

If you use these results, please cite:
```
@article{pot2024,
  title={Proof-of-Training: Black-Box Behavioral Verification of Neural Networks},
  author={...},
  journal={...},
  year={2024}
}
```
