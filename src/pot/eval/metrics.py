import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

def dist_logits_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def dist_kl(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    pa = softmax(a)
    pb = softmax(b)
    return float(np.sum(pa * (np.log(pa+eps) - np.log(pb+eps))))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def dist_text_edit(a: str, b: str) -> float:
    return 1.0 - SequenceMatcher(None, a, b).ratio()

def dist_text_embed(a_emb: np.ndarray, b_emb: np.ndarray) -> float:
    return float(1.0 - cosine_similarity(a_emb[None], b_emb[None])[0,0])