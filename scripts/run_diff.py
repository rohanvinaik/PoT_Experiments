from __future__ import annotations
import argparse
import os
import json
import yaml
from typing import Callable

from pot.verifier.core import (
    TestingMode,
    ModeParams,
    gen_challenge_seeds,
    iter_prompt_from_seed,
    EnhancedSequentialTester,
    Verdict,
)
from pot.verifier.logging import (
    write_summary_json,
    write_transcript_ndjson,
    pack_evidence_bundle,
)
from pot.verifier.lm import EchoModel, DummyAPIModel

# Optional HF import guarded
try:  # pragma: no cover - optional dependency
    from pot.verifier.lm import HFLocalModel
except Exception:  # noqa: BLE001
    HFLocalModel = None


def _mode_from_name(name: str) -> ModeParams:
    name = name.strip().upper()
    if name == "QUICK":
        return TestingMode.QUICK
    if name == "AUDIT":
        return TestingMode.AUDIT
    if name == "EXTENDED":
        return TestingMode.EXTENDED
    raise ValueError(f"Unknown mode: {name}")


def _make_model(spec: dict) -> object:
    t = spec.get("type")
    if t == "hf_local":
        if HFLocalModel is None:
            raise RuntimeError("Install HF extras: pip install '.[hf]'")
        return HFLocalModel(model_name=spec["model_name"])
    if t == "api":
        return DummyAPIModel(model_id=spec.get("model_id", "api-demo"))
    if t == "echo":
        return EchoModel(tag=spec.get("tag", "echo"))
    raise ValueError(f"Unknown model type: {t}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PoT behavioral diff test (sequential EB).")
    ap.add_argument("--config", required=True, help="YAML config")
    ap.add_argument("--outdir", default=None, help="Override output dir")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mode = _mode_from_name(cfg["mode"])
    run_id = cfg["run_id"]
    key_hex = cfg["hmac_key_hex"]
    n_chal = int(cfg.get("num_challenges", mode.n_max))
    outdir = args.outdir or cfg["output"]["dir"]
    os.makedirs(outdir, exist_ok=True)

    ref_model = _make_model(cfg["ref"])
    cand_model = _make_model(cfg["cand"])

    seeds = gen_challenge_seeds(key_hex=key_hex, run_id=run_id, n=n_chal)
    prompts = [iter_prompt_from_seed(s.seed_hex) for s in seeds]

    tester = EnhancedSequentialTester(params=mode)

    # Bind generator callables
    ref_generate: Callable[[str], str] = ref_model.generate
    cand_generate: Callable[[str], str] = cand_model.generate

    result = tester.run(prompts=prompts, ref_generate=ref_generate, cand_generate=cand_generate)

    # Write transcript + summary
    tpath = os.path.join(outdir, "transcript.ndjson")
    spath = os.path.join(outdir, "summary.json")
    epath = os.path.join(outdir, "evidence_bundle.zip")

    step_dicts = [vars(s) for s in result.steps]
    write_transcript_ndjson(tpath, step_dicts)
    write_summary_json(
        spath,
        {
            "run_id": run_id,
            "mode": cfg["mode"],
            "params": result.params,
            "verdict": result.verdict.value,
            "n_used": result.n_used,
            "ref_model": getattr(ref_model, "name", lambda: "unknown")(),
            "cand_model": getattr(cand_model, "name", lambda: "unknown")(),
            "challenge_count": n_chal,
            "outputs": {
                "transcript": os.path.basename(tpath),
                "summary": os.path.basename(spath),
            },
        },
    )
    pack_evidence_bundle(epath, files=[tpath, spath], extra_meta={"run_id": run_id})

    print(json.dumps({"verdict": result.verdict.value, "n_used": result.n_used, "outdir": outdir}, indent=2))


if __name__ == "__main__":
    main()