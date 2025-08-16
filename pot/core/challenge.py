from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import xxhash
import secrets
import hashlib
import os
import re
from .prf import prf_derive_key, prf_derive_seed, prf_floats, prf_integers, prf_choice, prf_bytes

@dataclass
class Challenge:
    """Individual challenge with unique ID and parameters."""
    challenge_id: str      # Unique identifier for this challenge
    index: int            # Challenge index in sequence
    family: str           # Challenge family (e.g., "vision:freq")
    parameters: Dict[str, Any]  # Challenge-specific parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "challenge_id": self.challenge_id,
            "index": self.index,
            "family": self.family,
            "parameters": self.parameters
        }

@dataclass
class ChallengeConfig:
    master_key_hex: str
    session_nonce_hex: str
    n: int
    family: str            # "vision:freq", "vision:texture", "lm:templates", ...
    params: Dict[str, Any] # family-specific args (freq ranges, token masks, etc.)
    model_id: Optional[str] = None  # Model identifier for challenge derivation

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

def generate_vision_freq_challenges(cfg: ChallengeConfig, seed: bytes, salt: str) -> List[Challenge]:
    """
    Generate sine grating challenges for the vision:freq family.
    
    Following the paper's specification, generates deterministic parameters:
    - frequency: cycles per degree (in specified range)
    - theta: orientation in degrees (0-180)
    - phase: phase offset in radians (0-2π)
    - contrast: Michelson contrast (0-1 or specified range)
    
    Each challenge has a unique ID: c_i = KDF(master_seed || model_id || i || salt)
    
    Args:
        cfg: Challenge configuration with freq_range and contrast_range
        seed: Deterministic seed derived from master key and model_id
        salt: Salt for commit-reveal protocol
    
    Returns:
        List of Challenge objects with sine grating parameters
    """
    challenges = []
    
    # Extract parameter ranges
    freq_range = cfg.params.get("freq_range", [0.1, 10.0])  # cycles per degree
    contrast_range = cfg.params.get("contrast_range", [0.1, 1.0])  # Michelson contrast
    
    for i in range(cfg.n):
        # Create unique derivation info for each challenge
        # Following paper: c_i = KDF(master_seed || model_id || i || salt)
        if cfg.model_id:
            challenge_info = f"challenge_{cfg.model_id}_{i}_{salt}".encode()
        else:
            challenge_info = f"challenge_{i}_{salt}".encode()
        
        # Generate deterministic parameters using PRF
        # Frequency in cycles per degree
        freq = prf_floats(seed, challenge_info + b":freq", 1, 
                         freq_range[0], freq_range[1])[0]
        
        # Orientation/theta in degrees (0-180)
        theta_deg = prf_floats(seed, challenge_info + b":theta", 1, 0, 180)[0]
        
        # Phase in radians (0-2π)
        phase_rad = prf_floats(seed, challenge_info + b":phase", 1, 0, 2 * np.pi)[0]
        
        # Contrast (Michelson contrast between 0 and 1)
        contrast = prf_floats(seed, challenge_info + b":contrast", 1,
                             contrast_range[0], contrast_range[1])[0]
        
        # Create unique challenge ID using xxhash
        # Hash all parameters plus index for uniqueness
        param_str = f"freq:{freq:.6f}_theta:{theta_deg:.6f}_phase:{phase_rad:.6f}_contrast:{contrast:.6f}_idx:{i}"
        challenge_id = xxhash.xxh3_64_hexdigest(param_str.encode() + seed[:8])
        
        # Create Challenge object with all parameters
        challenge = Challenge(
            challenge_id=challenge_id,
            index=i,
            family="vision:freq",
            parameters={
                "freq": float(freq),           # cycles per degree
                "theta": float(theta_deg),      # orientation in degrees
                "theta_rad": float(np.deg2rad(theta_deg)),  # also provide in radians
                "phase": float(phase_rad),      # phase in radians
                "contrast": float(contrast),    # Michelson contrast
                # Additional metadata
                "freq_range": freq_range,
                "contrast_range": contrast_range
            }
        )
        
        challenges.append(challenge)
    
    return challenges

def generate_vision_texture_challenges(cfg: ChallengeConfig, seed: bytes, salt: str) -> List[Challenge]:
    """
    Generate texture-based challenges for the vision:texture family.
    
    Following the paper's specification, generates deterministic texture patterns:
    - Perlin noise: multi-octave noise with persistence and scale
    - Gabor filters: oriented sinusoidal gratings with Gaussian envelope
    - Checkerboard: regular grid patterns with varying size and contrast
    
    Each challenge has a unique ID: c_i = KDF(master_seed || model_id || i || salt)
    
    Args:
        cfg: Challenge configuration with texture parameters
        seed: Deterministic seed derived from master key and model_id
        salt: Salt for commit-reveal protocol
    
    Returns:
        List of Challenge objects with texture parameters
    """
    challenges = []
    
    # Define available texture types
    texture_types = cfg.params.get("texture_types", ["perlin", "gabor", "checkerboard"])
    
    # Extract parameter ranges for each texture type
    # Perlin noise parameters
    perlin_params = cfg.params.get("perlin", {})
    octaves_range = perlin_params.get("octaves", [1, 5])  # Number of noise octaves
    persistence_range = perlin_params.get("persistence", [0.3, 0.7])  # Amplitude falloff
    scale_range = perlin_params.get("scale", [0.01, 0.1])  # Frequency scale
    
    # Gabor filter parameters
    gabor_params = cfg.params.get("gabor", {})
    wavelength_range = gabor_params.get("wavelength", [5.0, 30.0])  # Sinusoid wavelength in pixels
    orientation_range = gabor_params.get("orientation", [0, 180])  # Orientation in degrees
    phase_range = gabor_params.get("phase", [0, 2 * np.pi])  # Phase offset
    sigma_range = gabor_params.get("sigma", [3.0, 15.0])  # Gaussian envelope std dev
    aspect_ratio_range = gabor_params.get("aspect_ratio", [0.3, 1.0])  # Ellipticity
    
    # Checkerboard parameters
    checker_params = cfg.params.get("checkerboard", {})
    square_size_range = checker_params.get("square_size", [8, 32])  # Size of squares in pixels
    contrast_range = checker_params.get("contrast", [0.3, 1.0])  # Contrast level
    rotation_range = checker_params.get("rotation", [0, 45])  # Rotation in degrees
    
    for i in range(cfg.n):
        # Create unique derivation info for each challenge
        # Following paper: c_i = KDF(master_seed || model_id || i || salt)
        if cfg.model_id:
            challenge_info = f"challenge_{cfg.model_id}_{i}_{salt}".encode()
        else:
            challenge_info = f"challenge_{i}_{salt}".encode()
        
        # Deterministically select texture type
        texture_type = prf_choice(seed, challenge_info + b":texture_type", texture_types)
        
        # Generate parameters based on texture type
        if texture_type == "perlin":
            # Generate Perlin noise parameters
            octaves = prf_integers(seed, challenge_info + b":octaves", 1,
                                  octaves_range[0], octaves_range[1])[0]
            persistence = prf_floats(seed, challenge_info + b":persistence", 1,
                                   persistence_range[0], persistence_range[1])[0]
            scale = prf_floats(seed, challenge_info + b":scale", 1,
                             scale_range[0], scale_range[1])[0]
            
            # Generate random seed for Perlin noise generation
            noise_seed = prf_integers(seed, challenge_info + b":noise_seed", 1, 0, 2**31)[0]
            
            parameters = {
                "texture_type": "perlin",
                "octaves": int(octaves),
                "persistence": float(persistence),
                "scale": float(scale),
                "seed": int(noise_seed),
                # Metadata
                "octaves_range": octaves_range,
                "persistence_range": persistence_range,
                "scale_range": scale_range
            }
            
        elif texture_type == "gabor":
            # Generate Gabor filter parameters
            wavelength = prf_floats(seed, challenge_info + b":wavelength", 1,
                                   wavelength_range[0], wavelength_range[1])[0]
            orientation_deg = prf_floats(seed, challenge_info + b":orientation", 1,
                                        orientation_range[0], orientation_range[1])[0]
            phase = prf_floats(seed, challenge_info + b":phase", 1,
                             phase_range[0], phase_range[1])[0]
            sigma = prf_floats(seed, challenge_info + b":sigma", 1,
                             sigma_range[0], sigma_range[1])[0]
            aspect_ratio = prf_floats(seed, challenge_info + b":aspect_ratio", 1,
                                     aspect_ratio_range[0], aspect_ratio_range[1])[0]
            
            parameters = {
                "texture_type": "gabor",
                "wavelength": float(wavelength),
                "orientation": float(orientation_deg),  # in degrees
                "orientation_rad": float(np.deg2rad(orientation_deg)),  # also in radians
                "phase": float(phase),
                "sigma": float(sigma),
                "aspect_ratio": float(aspect_ratio),
                # Metadata
                "wavelength_range": wavelength_range,
                "orientation_range": orientation_range,
                "phase_range": phase_range,
                "sigma_range": sigma_range,
                "aspect_ratio_range": aspect_ratio_range
            }
            
        elif texture_type == "checkerboard":
            # Generate checkerboard parameters
            square_size = prf_integers(seed, challenge_info + b":square_size", 1,
                                      square_size_range[0], square_size_range[1])[0]
            contrast = prf_floats(seed, challenge_info + b":contrast", 1,
                                contrast_range[0], contrast_range[1])[0]
            rotation_deg = prf_floats(seed, challenge_info + b":rotation", 1,
                                    rotation_range[0], rotation_range[1])[0]
            
            # Generate phase offset for shifted checkerboards
            phase_x = prf_integers(seed, challenge_info + b":phase_x", 1, 0, square_size)[0]
            phase_y = prf_integers(seed, challenge_info + b":phase_y", 1, 0, square_size)[0]
            
            parameters = {
                "texture_type": "checkerboard",
                "square_size": int(square_size),
                "contrast": float(contrast),
                "rotation": float(rotation_deg),  # in degrees
                "rotation_rad": float(np.deg2rad(rotation_deg)),  # also in radians
                "phase_x": int(phase_x),
                "phase_y": int(phase_y),
                # Metadata
                "square_size_range": square_size_range,
                "contrast_range": contrast_range,
                "rotation_range": rotation_range
            }
            
        else:
            # Fallback to simple Perlin noise for unknown types
            octaves = prf_integers(seed, challenge_info + b":octaves", 1, 2, 4)[0]
            scale = prf_floats(seed, challenge_info + b":scale", 1, 0.02, 0.08)[0]
            
            parameters = {
                "texture_type": texture_type,
                "octaves": int(octaves),
                "scale": float(scale)
            }
        
        # Create unique challenge ID by hashing all parameters
        param_str = f"type:{texture_type}_params:{sorted(parameters.items())}_idx:{i}"
        challenge_id = xxhash.xxh3_64_hexdigest(param_str.encode() + seed[:8])
        
        # Create Challenge object
        challenge = Challenge(
            challenge_id=challenge_id,
            index=i,
            family="vision:texture",
            parameters=parameters
        )
        
        challenges.append(challenge)
    
    return challenges

def generate_lm_templates_challenges(cfg: ChallengeConfig, seed: bytes, salt: str) -> List[Challenge]:
    """
    Generate template-based text challenges for the lm:templates family.
    
    Following the paper's specification, generates deterministic prompts:
    - Templates with slots for grammatical components
    - Subjects, verbs, objects, modifiers/adjectives
    - Complete prompts like "The [adjective] [subject] [verb] the [object]"
    
    Each challenge has a unique ID: c_i = KDF(master_seed || model_id || i || salt)
    
    Args:
        cfg: Challenge configuration with templates and slot values
        seed: Deterministic seed derived from master key and model_id
        salt: Salt for commit-reveal protocol
    
    Returns:
        List of Challenge objects with templated prompts
    """
    challenges = []
    
    # Extract templates and slot values from params
    # Allow both old format (single templates list) and new format (categorized templates)
    if "templates" in cfg.params and isinstance(cfg.params["templates"], list):
        # Old format: list of template strings
        templates = cfg.params["templates"]
    else:
        # New format: default templates if not provided
        templates = cfg.params.get("templates", [
            "The {adjective} {subject} {verb} the {object}.",
            "{subject} {verb} {adjective} {object}.",
            "A {adjective} {subject} will {verb} the {object}.",
            "The {subject} {verb} {object} {adverb}.",
            "{adjective} {subject} {adverb} {verb} {object}.",
            "When the {subject} {verb}, the {object} becomes {adjective}.",
            "The {object} was {verb_past} by the {adjective} {subject}.",
            "{subject} and {subject2} {verb} the {adjective} {object}."
        ])
    
    # Extract slot values with defaults for common grammatical components
    slots = cfg.params.get("slots", {})
    
    # Provide comprehensive defaults if not specified
    default_slots = {
        "subject": ["cat", "dog", "bird", "robot", "scientist", "artist", "teacher", "student"],
        "subject2": ["mouse", "rabbit", "engineer", "doctor", "writer", "musician"],
        "verb": ["chases", "observes", "creates", "discovers", "analyzes", "transforms", "inspects", "measures"],
        "verb_past": ["created", "discovered", "analyzed", "transformed", "observed", "measured"],
        "object": ["ball", "puzzle", "painting", "equation", "melody", "story", "experiment", "pattern"],
        "adjective": ["clever", "curious", "colorful", "mysterious", "elegant", "complex", "simple", "unusual"],
        "adverb": ["quickly", "carefully", "quietly", "gracefully", "methodically", "suddenly", "slowly", "precisely"]
    }
    
    # Merge provided slots with defaults (deterministically)
    # Create new dict to avoid modifying input
    merged_slots = {}
    # First add all provided slots
    for key in sorted(slots.keys()):
        merged_slots[key] = slots[key]
    # Then add defaults for missing slots
    for key in sorted(default_slots.keys()):
        if key not in merged_slots:
            merged_slots[key] = default_slots[key]
    slots = merged_slots
    
    for i in range(cfg.n):
        # Create unique derivation info for each challenge
        # Following paper: c_i = KDF(master_seed || model_id || i || salt)
        if cfg.model_id:
            challenge_info = f"challenge_{cfg.model_id}_{i}_{salt}".encode()
        else:
            challenge_info = f"challenge_{i}_{salt}".encode()
        
        # Deterministically select a template
        template = prf_choice(seed, challenge_info + b":template", templates)
        
        # Extract slot names from the template
        slot_pattern = r'\{([^}]+)\}'
        required_slots = re.findall(slot_pattern, template)
        
        # Deterministically select values for each slot
        slot_values = {}
        for slot_name in required_slots:
            if slot_name in slots:
                # Use PRF to deterministically select from available values
                slot_value = prf_choice(seed, challenge_info + f":slot_{slot_name}".encode(), 
                                       slots[slot_name])
                slot_values[slot_name] = slot_value
            else:
                # If slot not defined, use a placeholder
                slot_values[slot_name] = f"[{slot_name}]"
        
        # Generate the complete prompt by filling in the template
        prompt = template
        for slot_name, slot_value in slot_values.items():
            prompt = prompt.replace(f"{{{slot_name}}}", slot_value)
        
        # Create unique challenge ID by hashing the complete prompt
        # This ensures identical prompts always get the same ID
        prompt_bytes = prompt.encode('utf-8')
        challenge_id = xxhash.xxh3_64_hexdigest(prompt_bytes + seed[:8])
        
        # Create Challenge object with all information
        challenge = Challenge(
            challenge_id=challenge_id,
            index=i,
            family="lm:templates",
            parameters={
                "template": template,
                "slot_values": slot_values,
                "prompt": prompt,
                # Additional metadata
                "available_slots": sorted(slots.keys()),  # Sort for determinism
                "template_index": templates.index(template) if template in templates else -1
            }
        )
        
        challenges.append(challenge)
    
    return challenges

def generate_challenges(cfg: ChallengeConfig) -> Dict[str, Any]:
    """
    Generate challenges using PRF-based derivation for cryptographic security.
    
    Implements the paper's specification:
    c_i = KDF(master_seed || model_id || i || salt)
    
    Returns:
      dict with keys:
        'challenge_id' : hex string (overall challenge set ID)
        'family'       : cfg.family
        'items'        : list of serialized challenge items
        'challenges'   : list of Challenge objects
        'salt'         : hex string used for commit-reveal
    """
    # Use PRF to derive seed with family, params, and model_id mixed in
    master_key = bytes.fromhex(cfg.master_key_hex)
    nonce = bytes.fromhex(cfg.session_nonce_hex)
    
    # Include model_id in the seed derivation if provided
    if cfg.model_id:
        # Mix model_id into params for seed derivation
        params_with_model = {**cfg.params, "model_id": cfg.model_id}
        seed = prf_derive_seed(master_key, cfg.family, params_with_model, nonce)
    else:
        seed = prf_derive_seed(master_key, cfg.family, cfg.params, nonce)
    
    # Generate deterministic salt from PRF
    salt_key = prf_derive_key(master_key, "salt", nonce)
    salt = salt_key[:16].hex()
    
    # Generate challenges based on family
    if cfg.family == "vision:freq":
        challenges = generate_vision_freq_challenges(cfg, seed, salt)
    elif cfg.family == "vision:texture":
        challenges = generate_vision_texture_challenges(cfg, seed, salt)
    elif cfg.family == "lm:templates":
        challenges = generate_lm_templates_challenges(cfg, seed, salt)
    else:
        # Fallback to old method for other families
        items = _sample_family_prf(cfg.family, cfg.params, cfg.n, seed)
        # Create Challenge objects for backward compatibility
        challenges = []
        for i, item in enumerate(items):
            # Generate unique challenge ID for each item
            challenge_bytes = prf_bytes(seed, f"challenge_{i}".encode(), 16)
            challenge_id = xxhash.xxh3_64_hexdigest(challenge_bytes)
            challenges.append(Challenge(
                challenge_id=challenge_id,
                index=i,
                family=cfg.family,
                parameters=item
            ))
    
    # Extract items for backward compatibility
    items = [c.parameters for c in challenges]
    
    # Generate overall challenge set ID
    cid = xxhash.xxh3_128_hexdigest(repr(items).encode() + bytes.fromhex(salt))
    
    return {
        "challenge_id": cid,
        "family": cfg.family,
        "items": items,
        "challenges": challenges,
        "salt": salt
    }

def _sample_family_prf(family: str, params: Dict[str, Any], n: int, seed: bytes):
    """Sample challenges using PRF for cryptographic security."""
    if family == "vision:freq":
        # sine gratings: frequency, orientation, phase, contrast
        # Note: This is kept for backward compatibility
        # New code should use generate_vision_freq_challenges
        items = []
        for i in range(n):
            # Use different info for each item to ensure independence
            item_info = f"item_{i}".encode()
            
            # Generate parameters using PRF
            freq_vals = prf_floats(seed, item_info + b":freq", 1, 
                                   params["freq_range"][0], params["freq_range"][1])
            # Keep theta in radians for backward compatibility
            theta_vals = prf_floats(seed, item_info + b":theta", 1, 0, np.pi)
            phase_vals = prf_floats(seed, item_info + b":phase", 1, 0, 2*np.pi)
            contrast_vals = prf_floats(seed, item_info + b":contrast", 1,
                                       params["contrast_range"][0], params["contrast_range"][1])
            
            items.append({
                "freq": float(freq_vals[0]),
                "theta": float(theta_vals[0]),  # radians for backward compatibility
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