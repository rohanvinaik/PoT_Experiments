import hmac
import hashlib
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ChallengeSeed:
    index: int
    seed_hex: str  # HMAC(K, run_id || i) hex


def _hmac_hex(key: bytes, msg: bytes) -> str:
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def gen_challenge_seeds(key_hex: str, run_id: str, n: int) -> list[ChallengeSeed]:
    """
    Deterministic, pre-committed challenge seeds.
    key_hex: secret hex string (store separately; publish HMAC of the seed list)
    run_id:  public run identifier
    """
    key = bytes.fromhex(key_hex)
    seeds: list[ChallengeSeed] = []
    for i in range(1, n + 1):
        seed_hex = _hmac_hex(key, f"{run_id}|{i}".encode("utf-8"))
        seeds.append(ChallengeSeed(index=i, seed_hex=seed_hex))
    return seeds


def iter_prompt_from_seed(seed_hex: str) -> str:
    """
    Minimal deterministic prompt generator from a seed.
    Replace with your real prompt family logic (stratified templates, etc.)
    """
    # Toy prompt family: pick a simple task keyed by first byte
    which = int(seed_hex[:2], 16) % 3
    if which == 0:
        return f"Summarize in one sentence: seed={seed_hex[:12]}"
    elif which == 1:
        return f"Compute 23*19 and explain briefly. seed={seed_hex[:12]}"
    else:
        return f"Define 'entropy' in 2 lines. seed={seed_hex[:12]}"