import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from pot.lm.hash_benchmark import run_benchmark


def test_fuzzy_beats_exact():
    metrics = run_benchmark()
    assert metrics["fuzzy"]["auroc"] >= metrics["exact"]["auroc"]
    for method in ["fuzzy", "exact"]:
        for key in ["auroc", "far", "frr"]:
            assert key in metrics[method]
