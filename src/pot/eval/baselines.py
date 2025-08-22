import hashlib
import json
import numpy as np

# Baseline verification methods


def naive_io_hash(outputs):
    """Naive I/O hash baseline"""
    return hashlib.sha256(str(outputs).encode()).hexdigest()


def statistical_fingerprint(outputs, quantiles=(0.25, 0.5, 0.75)):
    """Compute a statistical fingerprint of the given outputs."""
    arr = np.asarray(outputs, dtype=np.float64).ravel()
    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "quantiles": [float(q) for q in np.quantile(arr, quantiles)],
    }
    serialized = json.dumps(stats, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def lightweight_fingerprint(model, challenges, quantiles=(0.25, 0.5, 0.75)):
    """Generate a statistical fingerprint for model outputs."""
    outputs = [model(c) for c in challenges]
    return statistical_fingerprint(outputs, quantiles=quantiles)


def benign_input_fingerprint(outputs):
    """Benign-input fingerprint baseline (FBI)."""
    return statistical_fingerprint(outputs)


def adversarial_trajectory_fingerprint(trajectory):
    """Fingerprint derived from an adversarial trajectory of outputs."""
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    diffs = np.diff(arr, axis=0)
    stats = {
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs)),
    }
    serialized = json.dumps(stats, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def fixed_n_aggregation_distance(outputs1, outputs2, n=256, metric="l2"):
    """Distance between first n outputs using L2 or Hamming metrics."""
    arr1 = np.asarray(outputs1, dtype=np.float64).flatten()[:n]
    arr2 = np.asarray(outputs2, dtype=np.float64).flatten()[:n]

    if metric == "hamming":
        arr1 = (arr1 > 0.5).astype(np.int8)
        arr2 = (arr2 > 0.5).astype(np.int8)
        distance = np.mean(arr1 != arr2)
    else:  # default to L2
        diff = arr1 - arr2
        distance = np.linalg.norm(diff, ord=2) / max(len(arr1), 1)

    return float(distance)


def sequential_hoeffding_test(outputs1, outputs2, epsilon=0.05, bound=1.0):
    """Sequential Hoeffding test returning decision and queries used."""
    arr1 = np.asarray(outputs1, dtype=np.float64).flatten()
    arr2 = np.asarray(outputs2, dtype=np.float64).flatten()

    n_queries = 0
    cumulative = 0.0
    decision = "accept"

    for o1, o2 in zip(arr1, arr2):
        n_queries += 1
        diff = np.clip(o1 - o2, -bound, bound)
        cumulative += diff
        mean = cumulative / n_queries
        threshold = np.sqrt(0.5 * np.log(2 / epsilon) / n_queries)
        if abs(mean) > threshold:
            decision = "reject"
            break

    return decision, n_queries

