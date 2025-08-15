import pytest
from pot.lm.verifier import LMVerifier
from typing import Dict


class DummyTokenizer:
    pad_token_id = None
    bos_token_id = None
    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]


class DummyLM:
    def __init__(self, outputs: Dict[str, str]):
        self.outputs = outputs
        self.tok = DummyTokenizer()

    def generate(self, prompt: str, max_new_tokens: int = 64):
        return self.outputs.get(prompt, "")


class DummyVerifier(LMVerifier):
    def compute_output_distance(self, output1: str, output2: str, method: str = "fuzzy") -> float:
        if output1 == output2:
            return 0.0
        if output1 == "close" and output2 == "ref":
            return 0.12
        if output1 == "far" and output2 == "ref":
            return 0.2
        return 1.0


def test_verify_with_time_tolerance_nonlinear_cap():
    reference = DummyLM({"p": "ref"})
    verifier = DummyVerifier(reference)
    challenges = [{"prompt": "p"}]

    close_model = DummyLM({"p": "close"})
    far_model = DummyLM({"p": "far"})

    result_day0 = verifier.verify_with_time_tolerance(
        close_model,
        challenges,
        base_tolerance=0.1,
        days_elapsed=0,
        drift_rate=0.1,
        drift_model="quadratic",
        max_tolerance=0.15,
    )
    assert not result_day0.accepted
    assert "quadratic" in result_day0.metadata["time_tolerance"]["justification"]

    result_close_day2 = verifier.verify_with_time_tolerance(
        close_model,
        challenges,
        base_tolerance=0.1,
        days_elapsed=2,
        drift_rate=0.1,
        drift_model="quadratic",
        max_tolerance=0.15,
    )
    assert result_close_day2.accepted

    result_far_day2 = verifier.verify_with_time_tolerance(
        far_model,
        challenges,
        base_tolerance=0.1,
        days_elapsed=2,
        drift_rate=0.1,
        drift_model="quadratic",
        max_tolerance=0.15,
    )
    assert not result_far_day2.accepted
