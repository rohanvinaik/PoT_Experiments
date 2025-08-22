import math
from dataclasses import dataclass


@dataclass
class Welford:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of diffs from current mean

    def push(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)


def eb_halfwidth(var: float, n: int, delta: float) -> float:
    """
    Empirical-Bernstein half-width (bounded in [0,1] scores).
    U_n(delta) = sqrt(2 * var * log(1/delta) / n) + 7*log(1/delta) / (3*(n-1))
    """
    if n <= 1:
        return float("inf")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0,1)")
    log_term = math.log(1.0 / delta)
    a = math.sqrt(max(0.0, 2.0 * var * log_term / n))
    b = 7.0 * log_term / (3.0 * (n - 1))
    return a + b


def spending_schedule(alpha: float, n: int) -> float:
    """
    Simple conservative schedule for anytime bounds:
    delta_n = alpha / (n * (n + 1))    (sum_n delta_n <= alpha)
    """
    return alpha / (n * (n + 1))