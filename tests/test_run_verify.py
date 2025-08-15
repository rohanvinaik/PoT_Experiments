import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def make_model(weight: float) -> torch.nn.Module:
    linear = torch.nn.Linear(4, 2)
    with torch.no_grad():
        linear.weight.fill_(weight)
        linear.bias.zero_()
    return linear


def test_run_verify_end_to_end(tmp_path):
    ref_path = tmp_path / "ref.pt"
    test_path = tmp_path / "test.pt"
    torch.save(make_model(1.0), ref_path)
    torch.save(make_model(1.1), test_path)

    config = {
        "experiment": "test_exp",
        "models": {
            "reference_path": str(ref_path),
            "test_path": str(test_path),
        },
        "challenges": {
            "families": [
                {
                    "family": "vision:freq",
                    "n": 2,
                    "params": {"freq_range": [0.5, 1.0], "contrast_range": [0.5, 0.5]},
                }
            ]
        },
        "verification": {"distances": ["logits_l2"], "tau_grid": [0.05]},
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)

    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_verify.py",
            "--config",
            str(cfg_path),
            "--challenge_family",
            "vision:freq",
            "--n",
            "2",
            "--output_dir",
            str(out_dir),
        ],
        check=True,
    )

    run_path = out_dir / "test_exp" / "vision:freq_2"
    distances_file = run_path / "distances.jsonl"
    verify_file = run_path / "verify.jsonl"

    assert distances_file.exists()
    assert verify_file.exists()

    distances_lines = list(open(distances_file))
    assert len(distances_lines) == 2

    summary = [json.loads(line) for line in open(verify_file)]
    assert summary and "far_hat" in summary[0]

