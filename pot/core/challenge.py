from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import xxhash
import secrets
import hashlib
import os
from .prf import prf_derive_key, prf_derive_seed, prf_floats, prf_integers, prf_choice

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
    Now uses PRF module for consistency
    """
    master_key = bytes.fromhex(master_key_hex)
    nonce = bytes.fromhex(nonce_hex)
    return prf_derive_key(master_key, label, nonce)

def seeded_rng(seed: bytes) -> np.random.Generator:
    s = int.from_bytes(seed, "big") % (2**63 - 1)
    return np.random.default_rng(s)

def generate_challenges(cfg: ChallengeConfig) -> Dict[str, Any]:
    """
    Generate challenges using PRF-based derivation for cryptographic security.
    
    Returns:
      dict with keys:
        'challenge_id' : hex string
        'family'       : cfg.family
        'items'        : list of serialized challenge items
        'salt'         : hex string used for commit-reveal
    """
    # Use PRF to derive seed with family and params mixed in
    master_key = bytes.fromhex(cfg.master_key_hex)
    nonce = bytes.fromhex(cfg.session_nonce_hex)
    seed = prf_derive_seed(master_key, cfg.family, cfg.params, nonce)
    
    # Option 1: Use PRF directly for sampling (more secure)
    items = _sample_family_prf(cfg.family, cfg.params, cfg.n, seed)
    
    # Generate deterministic salt from PRF (instead of secrets.token_hex)
    salt_key = prf_derive_key(master_key, "salt", nonce)
    salt = salt_key[:16].hex()
    
    cid = xxhash.xxh3_128_hexdigest(repr(items).encode() + bytes.fromhex(salt))
    return {"challenge_id": cid, "family": cfg.family, "items": items, "salt": salt}

def _sample_family_prf(family: str, params: Dict[str, Any], n: int, seed: bytes):
    """Sample challenges using PRF for cryptographic security."""
    if family == "vision:freq":
        # sine gratings: frequency, orientation, phase, contrast
        items = []
        for i in range(n):
            # Use different info for each item to ensure independence
            item_info = f"item_{i}".encode()
            
            # Generate parameters using PRF
            freq_vals = prf_floats(seed, item_info + b":freq", 1, 
                                   params["freq_range"][0], params["freq_range"][1])
            theta_vals = prf_floats(seed, item_info + b":theta", 1, 0, np.pi)
            phase_vals = prf_floats(seed, item_info + b":phase", 1, 0, 2*np.pi)
            contrast_vals = prf_floats(seed, item_info + b":contrast", 1,
                                       params["contrast_range"][0], params["contrast_range"][1])
            
            items.append({
                "freq": float(freq_vals[0]),
                "theta": float(theta_vals[0]),
                "phase": float(phase_vals[0]),
                "contrast": float(contrast_vals[0])
            })
        return items
    
    if family == "vision:texture":
        # structured noise parameters
        items = []
        for i in range(n):
            item_info = f"item_{i}".encode()
            
            octave_vals = prf_integers(seed, item_info + b":octaves", 1,
                                       params["octaves"][0], params["octaves"][1])
            octave = octave_vals[0]
            
            scale_vals = prf_floats(seed, item_info + b":scale", 1,
                                    params["scale"][0], params["scale"][1])
            
            items.append({
                "octaves": int(octave),
                "scale": float(scale_vals[0])
            })
        return items
    
    if family == "lm:templates":
        # choose template types + perturbations
        items = []
        for i in range(n):
            item_info = f"item_{i}".encode()
            
            # Choose template
            template = prf_choice(seed, item_info + b":template", params["templates"])
            
            # Choose slot values
            slot_values = {}
            for k, v in params["slots"].items():
                slot_key = item_info + f":slot_{k}".encode()
                slot_values[k] = prf_choice(seed, slot_key, v)
            
            items.append({
                "template": template,
                "slot_values": slot_values
            })
        return items
    
    raise ValueError(f"Unknown family {family}")

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

def generate_challenges_legacy(cfg: ChallengeConfig) -> Dict[str, Any]:
    """
    Legacy challenge generation using numpy RNG (kept for compatibility).
    """
    seed = kdf(cfg.master_key_hex, "challenge:"+cfg.family, cfg.session_nonce_hex)
    rng = seeded_rng(seed)
    salt = secrets.token_hex(16)
    items = _sample_family(cfg.family, cfg.params, cfg.n, rng)
    cid = xxhash.xxh3_128_hexdigest(repr(items).encode() + bytes.fromhex(salt))
    return {"challenge_id": cid, "family": cfg.family, "items": items, "salt": salt}