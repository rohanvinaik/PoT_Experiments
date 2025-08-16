import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pot.eval.baselines import (
    statistical_fingerprint,
    benign_input_fingerprint,
    adversarial_trajectory_fingerprint,
    fixed_n_aggregation_distance,
    sequential_hoeffding_test,
)


def test_statistical_fingerprint_identical():
    outputs1 = np.array([0.1, 0.2, 0.3])
    outputs2 = np.array([0.1, 0.2, 0.3])
    assert statistical_fingerprint(outputs1) == statistical_fingerprint(outputs2)


def test_statistical_fingerprint_perturbed():
    outputs = np.array([0.1, 0.2, 0.3])
    perturbed = outputs.copy()
    perturbed[0] += 0.01
    assert statistical_fingerprint(outputs) != statistical_fingerprint(perturbed)


def test_benign_input_fingerprint():
    outputs1 = np.array([0.4, 0.5, 0.6])
    outputs2 = np.array([0.4, 0.5, 0.6])
    assert benign_input_fingerprint(outputs1) == benign_input_fingerprint(outputs2)


def test_adversarial_trajectory_fingerprint():
    traj1 = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
    traj2 = np.array([[0.1, 0.2], [0.2, 0.3], [0.31, 0.4]])
    assert adversarial_trajectory_fingerprint(traj1) != adversarial_trajectory_fingerprint(traj2)


def test_fixed_n_aggregation_distance_hamming():
    out1 = np.array([0, 1, 1, 0])
    out2 = np.array([0, 0, 1, 0])
    assert fixed_n_aggregation_distance(out1, out2, n=4, metric="hamming") == 0.25


def test_sequential_hoeffding_accept():
    out1 = [0.0] * 20
    out2 = [0.0] * 20
    decision, n_q = sequential_hoeffding_test(out1, out2, epsilon=0.05)
    assert decision == "accept"
    assert n_q == 20


def test_sequential_hoeffding_reject():
    out1 = [1.0] * 20
    out2 = [0.0] * 20
    decision, _ = sequential_hoeffding_test(out1, out2, epsilon=0.05)
    assert decision == "reject"

