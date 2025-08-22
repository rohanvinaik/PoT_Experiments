from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Callable, Iterable

from .modes import ModeParams
from .stats import Welford, eb_halfwidth, spending_schedule
from .scoring import bounded_difference


class Verdict(str, Enum):
    SAME = "SAME"
    DIFFERENT = "DIFFERENT"
    UNDECIDED = "UNDECIDED"


@dataclass
class StepRecord:
    index: int
    prompt: str
    ref_output: str
    cand_output: str
    score: float
    mean: float
    var: float
    halfwidth: float
    delta_n: float
    verdict_so_far: str


@dataclass
class RunResult:
    verdict: Verdict
    steps: list[StepRecord]
    n_used: int
    params: dict


class EnhancedSequentialTester:
    """
    Minimal sequential tester:
    - Streams per-prompt difference scores (in [0,1]).
    - Maintains Welford stats and EB half-width with a spending schedule.
    - SAME if CI entirely in [-gamma, +gamma] and narrow enough.
    - DIFFERENT if |mean| >= delta* and relative margin <= eps_diff.
      (Here mean is >=0, since scores are differences; we keep the absolute check for template parity.)
    """

    def __init__(
        self,
        params: ModeParams,
        score_fn: Callable[[str, str], float] = bounded_difference,
    ) -> None:
        self.params = params
        self.score_fn = score_fn

    def _same_rule(self, mean: float, h: float) -> bool:
        # Scores are >=0; SAME band around 0 means mean+h <= gamma and h <= eta*gamma
        return (mean + h) <= self.params.gamma and (h <= self.params.eta * self.params.gamma)

    def _diff_rule(self, mean: float, h: float) -> bool:
        # DIFFERENT if mean >= delta* and relative margin h/mean <= eps_diff
        if mean < self.params.delta_star:
            return False
        rme = h / max(mean, 1e-12)
        return rme <= self.params.eps_diff

    def run(
        self,
        prompts: Iterable[str],
        ref_generate: Callable[[str], str],
        cand_generate: Callable[[str], str],
    ) -> RunResult:
        w = Welford()
        steps: list[StepRecord] = []
        verdict: Verdict = Verdict.UNDECIDED

        for i, prompt in enumerate(prompts, start=1):
            # Generate outputs (should be deterministic decode)
            ref_out = ref_generate(prompt)
            cand_out = cand_generate(prompt)

            s = self.score_fn(ref_out, cand_out)
            w.push(s)

            delta_n = spending_schedule(self.params.alpha, w.n)
            h = eb_halfwidth(w.var, w.n, delta_n)

            # decision checks (respect n_min)
            if w.n >= self.params.n_min:
                if self._same_rule(w.mean, h):
                    verdict = Verdict.SAME
                elif self._diff_rule(w.mean, h):
                    verdict = Verdict.DIFFERENT
                else:
                    verdict = Verdict.UNDECIDED
            else:
                verdict = Verdict.UNDECIDED

            steps.append(
                StepRecord(
                    index=i,
                    prompt=prompt,
                    ref_output=ref_out,
                    cand_output=cand_out,
                    score=s,
                    mean=w.mean,
                    var=w.var,
                    halfwidth=h,
                    delta_n=delta_n,
                    verdict_so_far=verdict.value,
                )
            )

            if verdict in (Verdict.SAME, Verdict.DIFFERENT) or w.n >= self.params.n_max:
                break

        return RunResult(
            verdict=verdict,
            steps=steps,
            n_used=len(steps),
            params={
                **asdict(self.params),
                "note": "Empirical-Bernstein confidence sequence; HMAC-precommitted prompts upstream.",
            },
        )