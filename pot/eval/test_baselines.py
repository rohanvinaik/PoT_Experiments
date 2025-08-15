import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pot.eval.baselines import statistical_fingerprint


def test_statistical_fingerprint_identical():
    outputs1 = np.array([0.1, 0.2, 0.3])
    outputs2 = np.array([0.1, 0.2, 0.3])
    assert statistical_fingerprint(outputs1) == statistical_fingerprint(outputs2)


def test_statistical_fingerprint_perturbed():
    outputs = np.array([0.1, 0.2, 0.3])
    perturbed = outputs.copy()
    perturbed[0] += 0.01
    assert statistical_fingerprint(outputs) != statistical_fingerprint(perturbed)

