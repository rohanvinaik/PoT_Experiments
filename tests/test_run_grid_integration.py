import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
import pytest

from pot.core.challenge import ChallengeConfig, generate_challenges


def load_model(path: str):
    module_name, attr = path.split(":")
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)()


def compute_distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def test_run_grid_integration(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "tests/mock_config.yaml"
    output_dir = tmp_path / "out"

    cmd = [
        "python",
        str(repo_root / "scripts/run_grid.py"),
        "--config",
        str(config_path),
        "--exp",
        "TEST",
        "--output_dir",
        str(output_dir),
        "--n_values",
        "4",
    ]
    subprocess.run(cmd, check=True)

    result_file = output_dir / "mock_exp/TEST/grid_results.jsonl"
    assert result_file.exists()

    with open(result_file) as f:
        entries = [json.loads(line) for line in f]

    # Ensure variant pair was evaluated for each tau value
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    variant_entries = [e for e in entries if e["test_model"] == "variant"]
    assert len(variant_entries) == len(cfg["verification"]["tau_grid"])

    # Reconstruct challenges and compute mean distance for variant pair
    chal_cfg = ChallengeConfig(
        master_key_hex=cfg["challenges"]["master_key"],
        session_nonce_hex=cfg["challenges"]["session_nonce"],
        n=4,
        family=cfg["challenges"]["family"],
        params=cfg["challenges"]["params"],
    )
    challenges = generate_challenges(chal_cfg)["items"]

    ref_model = load_model(cfg["models"]["reference"])
    var_model = load_model(cfg["models"]["variant"])
    distances = [compute_distance(ref_model(ch), var_model(ch)) for ch in challenges]
    mean_distance = float(np.mean(distances))
    assert variant_entries[0]["mean_distance"] == pytest.approx(mean_distance)

    # Verify AUROC computed from FAR/FRR
    far = np.array([e["far"] for e in variant_entries])
    frr = np.array([e["frr"] for e in variant_entries])
    order = np.argsort(far)
    auroc = float(np.trapz(1 - frr[order], far[order]))
    assert variant_entries[0]["auroc"] == pytest.approx(auroc)

