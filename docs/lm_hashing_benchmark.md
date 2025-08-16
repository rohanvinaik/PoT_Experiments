# LM Hashing Benchmark

This benchmark compares fuzzy hashing with exact token matching on a small set of language model outputs.

| Method | AUROC | FAR | FRR |
|--------|------:|----:|----:|
| Fuzzy hashing | 0.875 | 0.0 | 0.5 |
| Exact match | 0.750 | 0.0 | 0.5 |

The ROC curves can be reproduced by running:

```
python -m pot.lm.hash_benchmark
```

This command saves an image to `docs/lm_hashing_roc.png` (not tracked in
this repository to avoid binary files).
