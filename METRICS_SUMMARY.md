# Experimental Metrics Summary

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

