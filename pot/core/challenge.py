from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import xxhash
import secrets
import hashlib
import os

@dataclass
class ChallengeConfig:
    master_key_hex: str
    session_nonce_hex: str
    n: int
    family: str            # "vision:freq", "vision:texture", "lm:templates", ...
    params: Dict[str, Any] # family-specific args (freq ranges, token masks, etc.)

def kdf(master_key_hex: str, label: str, nonce_hex: str) -> bytes:
    """
    Key Derivation Function for challenge generation
    Matches the KDF in governance.py for consistency with paper
    """
    import hmac
    master_key = bytes.fromhex(master_key_hex)
    context = bytes.fromhex(nonce_hex)
    info = label.encode() + context
    return hmac.new(master_key, info, hashlib.sha256).digest()

def seeded_rng(seed: bytes) -> np.random.Generator:
    s = int.from_bytes(seed, "big") % (2**63 - 1)
    return np.random.default_rng(s)

def generate_challenges(cfg: ChallengeConfig) -> Dict[str, Any]:
    """
    Returns:
      dict with keys:
        'challenge_id' : hex string
        'family'       : cfg.family
        'items'        : list of serialized challenge items
        'salt'         : hex string used for commit-reveal
    """
    seed = kdf(cfg.master_key_hex, "challenge:"+cfg.family, cfg.session_nonce_hex)
    rng = seeded_rng(seed)
    salt = secrets.token_hex(16)
    items = _sample_family(cfg.family, cfg.params, cfg.n, rng)
    cid = xxhash.xxh3_128_hexdigest(repr(items).encode() + bytes.fromhex(salt))
    return {"challenge_id": cid, "family": cfg.family, "items": items, "salt": salt}

def _sample_family(family: str, params: Dict[str, Any], n: int, rng: np.random.Generator):
    if family == "vision:freq":
        # sine gratings: frequency, orientation, phase, contrast
        return [{"freq": float(rng.uniform(*params["freq_range"])),
                 "theta": float(rng.uniform(0, np.pi)),
                 "phase": float(rng.uniform(0, 2*np.pi)),
                 "contrast": float(rng.uniform(*params["contrast_range"]))} for _ in range(n)]
    if family == "vision:texture":
        # structured noise parameters
        return [{"octaves": int(rng.integers(*params["octaves"])),
                 "scale": float(rng.uniform(*params["scale"]))} for _ in range(n)]
    if family == "lm:templates":
        # choose template types + perturbations
        return [{"template": rng.choice(params["templates"]),
                 "slot_values": {k: rng.choice(v) for k, v in params["slots"].items()}}
                 for _ in range(n)]
    raise ValueError(f"Unknown family {family}")