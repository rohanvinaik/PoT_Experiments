import numpy as np
import re

def canonicalize_logits(logits: np.ndarray, p: int = 6, eps: float = 1e-6):
    q = np.round(logits, p)
    q[np.abs(q) < eps] = 0.0
    return q

_whitespace_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s]")

def canonicalize_text(s: str, lower=True, strip_punct=True, collapse_ws=True, max_len=None):
    if lower: s = s.lower()
    if strip_punct: s = _punct_re.sub("", s)
    if collapse_ws: s = _whitespace_re.sub(" ", s).strip()
    if max_len is not None: s = s[:max_len]
    return s