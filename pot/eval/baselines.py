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

