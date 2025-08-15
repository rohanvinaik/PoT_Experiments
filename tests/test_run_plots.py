import json
import os
import subprocess
import sys
from pathlib import Path


def test_leakage_plot_generation(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    log_file = exp_dir / "leakage_log.jsonl"
    entries = [
        {"rho": 0.0, "detection_rate": 1.0},
        {"rho": 0.25, "detection_rate": 0.8},
    ]
    with open(log_file, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_plots.py",
            "--exp_dir",
            str(exp_dir),
            "--plot_type",
            "leakage",
            "--input_files",
            str(log_file),
        ],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert (exp_dir / "leakage_curve.png").exists()
