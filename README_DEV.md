# PoT Verifier — Developer Notes (Skeleton)

This is a minimal, runnable scaffold:
- HMAC challenge generation (pre-commitment).
- Empirical-Bernstein (EB) confidence sequence.
- SAME/DIFFERENT decision rules with early stopping.
- Transcript logging + evidence bundle (zip).
- Stubs for local HF models and API models.

Run:
```bash
python -m scripts.run_diff --config configs/example_local.yaml
```

Optional HF install:
```bash
pip install ".[hf]"
```

---

## Core modules

### `pot/verifier/core/modes.py`

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeParams:
    alpha: float
    gamma: float
    eta: float
    delta_star: float
    eps_diff: float
    n_min: int
    n_max: int


class TestingMode:
    QUICK = ModeParams(alpha=0.025, gamma=0.15, eta=0.50, delta_star=0.80, eps_diff=0.15, n_min=10, n_max=120)
    AUDIT = ModeParams(alpha=0.010, gamma=0.10, eta=0.50, delta_star=1.00, eps_diff=0.10, n_min=30, n_max=400)
    EXTENDED = ModeParams(alpha=0.001, gamma=0.08, eta=0.40, delta_star=1.10, eps_diff=0.08, n_min=50, n_max=800)
```