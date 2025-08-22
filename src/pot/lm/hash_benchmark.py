import json
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .fuzzy_hash import TokenSpaceNormalizer


class SimpleTokenizer:
    """Minimal whitespace tokenizer with deterministic ID mapping."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = text.lower().replace('.', '').split()
        ids = []
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab) + 1
            ids.append(self.vocab[tok])
        return ids


def _far_frr(distances: List[float], labels: List[int], threshold: float = 0.5) -> Tuple[float, float]:
    """Compute false accept/reject rates for given distances and labels."""
    d = np.array(distances)
    y = np.array(labels)
    accept = d < threshold
    if np.any(y == 0):
        far = np.mean(accept[y == 0])
    else:
        far = 0.0
    if np.any(y == 1):
        frr = np.mean(~accept[y == 1])
    else:
        frr = 0.0
    return float(far), float(frr)


def run_benchmark(plot_path: str | None = None) -> Dict[str, Dict[str, float]]:
    """Run fuzzy vs exact hashing benchmark on sample LM outputs.

    Args:
        plot_path: Optional path to save ROC plot.

    Returns:
        Dictionary with metrics for fuzzy and exact methods.
    """
    pairs: List[Tuple[str, str, int]] = [
        ("the cat sat on the mat", "the cat sat on the mat", 1),
        ("the cat sat on the mat", "on the mat sat the cat", 1),
        ("the cat sat on the mat", "the dog sat on the rug", 0),
        ("hello world", "hello world", 1),
        ("i love eating pizza", "i hate eating pizza", 0),
        ("the sky is blue", "the sky was blue", 1),
        ("good morning", "good night", 0),
    ]

    tokenizer = SimpleTokenizer()
    normalizer = TokenSpaceNormalizer(tokenizer)

    distances_exact: List[float] = []
    distances_fuzzy: List[float] = []
    labels: List[int] = []

    for a, b, label in pairs:
        tokens_a = tokenizer.encode(a, add_special_tokens=False)
        tokens_b = tokenizer.encode(b, add_special_tokens=False)
        distances_exact.append(normalizer.compute_distance(tokens_a, tokens_b, method="exact"))
        distances_fuzzy.append(normalizer.compute_distance(tokens_a, tokens_b, method="fuzzy"))
        labels.append(label)

    scores_exact = [1 - d for d in distances_exact]
    scores_fuzzy = [1 - d for d in distances_fuzzy]

    auc_exact = float(roc_auc_score(labels, scores_exact))
    auc_fuzzy = float(roc_auc_score(labels, scores_fuzzy))

    far_exact, frr_exact = _far_frr(distances_exact, labels)
    far_fuzzy, frr_fuzzy = _far_frr(distances_fuzzy, labels)

    if plot_path is not None:
        import matplotlib.pyplot as plt

        fpr_e, tpr_e, _ = roc_curve(labels, scores_exact)
        fpr_f, tpr_f, _ = roc_curve(labels, scores_fuzzy)
        plt.figure()
        plt.plot(fpr_e, tpr_e, label=f"Exact (AUROC={auc_exact:.2f})")
        plt.plot(fpr_f, tpr_f, label=f"Fuzzy (AUROC={auc_fuzzy:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Fuzzy vs Exact Matching")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return {
        "fuzzy": {"auroc": auc_fuzzy, "far": far_fuzzy, "frr": frr_fuzzy},
        "exact": {"auroc": auc_exact, "far": far_exact, "frr": frr_exact},
    }


if __name__ == "__main__":
    from pathlib import Path

    plot = Path(__file__).resolve().parents[2] / "docs" / "lm_hashing_roc.png"
    metrics = run_benchmark(str(plot))
    print(json.dumps(metrics, indent=2))
