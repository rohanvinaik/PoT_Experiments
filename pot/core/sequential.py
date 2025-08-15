from typing import Iterable, Callable
import numpy as np

# Simple SPRT-like skeleton; LLM can fill
def sprt_test(stream: Iterable[float], mu0: float, mu1: float, alpha: float, beta: float):
    """
    Sequential probability ratio test for Bernoulli-thresholded indicators or
    (simpler) additive e-values on centered distances.
    Returns 'accept_H0'|'accept_H1'|'continue' with running log and query count.
    """
    # TODO: implement proper likelihood updates or e-value martingale
    pass