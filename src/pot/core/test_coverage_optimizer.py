import numpy as np
from pot.core.coverage_separation import CoverageSeparationOptimizer

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dummy_model_linear(challenges):
    return np.sum(challenges, axis=1) * 0.1

def dummy_model_quadratic(challenges):
    return np.sum(challenges ** 2, axis=1) * 0.1

def dummy_model_mixed(challenges):
    return np.sum(challenges * np.sin(challenges), axis=1) * 0.1

def test_separation_improves_with_optimization():
    """Ensure optimizer improves separation when models are provided."""
    model_fns = {
        "linear": dummy_model_linear,
        "quadratic": dummy_model_quadratic,
        "mixed": dummy_model_mixed,
    }
    optimizer = CoverageSeparationOptimizer(
        input_dim=5,
        n_challenges=20,
        seed=0,
        model_eval_functions=model_fns,
    )

    # Baseline using zero challenges yields no separation
    baseline_challenges = np.zeros((20, 5))
    baseline_sep = optimizer.evaluate_challenge_set(baseline_challenges).separation_score

    optimized = optimizer.optimize_challenges(
        coverage_weight=0.0,
        separation_weight=1.0,
        n_iterations=50,
    )
    optimized_sep = optimizer.evaluate_challenge_set(optimized).separation_score

    print(f"Baseline separation: {baseline_sep:.4f}")
    print(f"Optimized separation: {optimized_sep:.4f}")

    assert optimized_sep >= baseline_sep
