import pytest

from wrapper_detection import WrapperAttackDetector


def test_perfect_response_and_quality():
    detector = WrapperAttackDetector()

    # Exact string match
    resp = {"output": "hello", "reference": "hello"}
    assert detector._is_perfect_response(resp)
    assert detector._response_quality(resp) == 1.0

    # Numeric near match within threshold
    near = {"output": 1.0000001, "reference": 1.0}
    assert detector._is_perfect_response(near)

    # Clearly different
    diff = {"output": 1.0, "reference": 2.0}
    assert not detector._is_perfect_response(diff)
    assert detector._response_quality(diff) == pytest.approx(0.5)


def test_challenge_memorization_detection():
    detector = WrapperAttackDetector(quality_drop_threshold=0.8)
    challenges = [{"id": i} for i in range(5)]
    responses = [{"output": 1.0, "reference": 1.0} for _ in range(5)]

    def eval_bad(_challenge):
        return {"output": 0.0, "reference": 1.0}

    score = detector.detect_challenge_memorization(challenges, responses, eval_bad)
    assert score == 1.0

    def eval_good(_challenge):
        return {"output": 1.0, "reference": 1.0}

    score = detector.detect_challenge_memorization(challenges, responses, eval_good)
    assert score == 0.0
