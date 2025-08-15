import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.run_attack import run_attack

CONFIG = ROOT / "configs" / "lm_small.yaml"


def test_wrapper_attack_metrics(tmp_path):
    entry = run_attack(str(CONFIG), "wrapper", rho=0.0, output_dir=str(tmp_path))
    metrics = entry["metrics"]
    assert metrics["training_steps"] == 0
    assert metrics["queries_used"] == 0
    assert metrics["attack_cost"] >= 0


def test_targeted_finetune_metrics_vary(tmp_path):
    entry1 = run_attack(str(CONFIG), "targeted_finetune", rho=0.1, output_dir=str(tmp_path))
    entry2 = run_attack(str(CONFIG), "targeted_finetune", rho=0.5, output_dir=str(tmp_path))
    assert entry1["metrics"]["attack_cost"] < entry2["metrics"]["attack_cost"]
    assert entry1["metrics"]["training_steps"] < entry2["metrics"]["training_steps"]


def test_distillation_metrics_vary(tmp_path):
    entry1 = run_attack(str(CONFIG), "distillation", budget=100, output_dir=str(tmp_path))
    entry2 = run_attack(str(CONFIG), "distillation", budget=200, output_dir=str(tmp_path))
    assert entry1["metrics"]["attack_cost"] < entry2["metrics"]["attack_cost"]
    assert entry1["metrics"]["queries_used"] == 100
    assert entry2["metrics"]["queries_used"] == 200
