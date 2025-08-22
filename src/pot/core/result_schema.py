import time
import torch
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import hashlib

def generate_merkle_root(challenges: List[str]) -> str:
    """Generate Merkle root for challenges"""
    if not challenges:
        return "0" * 64
    
    # Simple Merkle tree (can be made more sophisticated)
    hashes = [hashlib.sha256(c.encode()).hexdigest() for c in challenges]
    
    while len(hashes) > 1:
        next_level = []
        for i in range(0, len(hashes), 2):
            if i + 1 < len(hashes):
                combined = hashes[i] + hashes[i + 1]
            else:
                combined = hashes[i] + hashes[i]
            next_level.append(hashlib.sha256(combined.encode()).hexdigest())
        hashes = next_level
    
    return hashes[0]

def build_result(tester,
                config,
                info: Dict[str, Any],
                timing: Dict[str, float],
                challenges_used: List[str],
                hardware: Optional[str] = None) -> Dict[str, Any]:
    """Build comprehensive result with all fields"""
    
    # Determine hardware if not provided
    if hardware is None:
        if torch.cuda.is_available():
            hardware = "cuda"
        elif torch.backends.mps.is_available():
            hardware = "mps"
        else:
            hardware = "cpu"
    
    # Ensure info is not None
    if info is None:
        info = {"decision": "UNDECIDED", "rule": "", "ci": [0.0, 0.0], "half_width": 0.0}
    
    # Calculate RME
    if "rme" not in info:
        if "half_width" in info and hasattr(tester, "mean"):
            denom = max(abs(tester.mean), getattr(config, "min_effect_floor", 0.001))
            info["rme"] = info["half_width"] / denom
    
    result = {
        # Core decision
        "decision": info.get("decision", "UNDECIDED"),
        "rule": info.get("rule", ""),
        
        # Statistical parameters
        "alpha": config.alpha,
        "beta": getattr(config, "beta", config.alpha),
        "gamma": config.gamma,
        "delta_star": config.delta_star,
        
        # Sampling info
        "n_used": info.get("n_used", tester.n),
        "n_max": config.n_max,
        "n_eff": tester.n * config.positions_per_prompt,
        
        # Statistics
        "mean": float(tester.mean),
        "variance": float(getattr(tester, "variance", 0)),
        "ci_99": [float(info["ci"][0]), float(info["ci"][1])] if "ci" in info else None,
        "half_width": float(info.get("half_width", 0)),
        "rme": float(info.get("rme", 0)),
        
        # Configuration
        "positions_per_prompt": config.positions_per_prompt,
        "ci_method": config.ci_method,
        "mode": config.mode,
        
        # Timing
        "time": {
            "t_load": timing.get("t_load", 0),
            "t_infer_total": timing.get("t_infer_total", 0),
            "t_per_query": timing.get("t_per_query", 0),
            "t_total": timing.get("t_total", 0)
        },
        
        # Hardware
        "hardware": {
            "backend": hardware,
            "device": hardware
        },
        
        # Audit trail
        "challenge_namespace": getattr(config, "namespace", "default"),
        "merkle_root": generate_merkle_root(challenges_used),
        
        # Metadata
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "version": "1.0.0"
    }
    
    return result

def save_result(result: Dict[str, Any], 
                output_dir: Optional[Path] = None,
                filename: Optional[str] = None) -> Path:
    """Save result to JSON file"""
    
    if output_dir is None:
        output_dir = Path("outputs/results")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"result_{result['timestamp']}.json"
    
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return output_path

def validate_result(result: Dict[str, Any]) -> List[str]:
    """Validate result has all required fields"""
    
    required_fields = [
        "decision", "rule", "alpha", "beta", "n_used", "mean",
        "ci_99", "half_width", "rme", "positions_per_prompt",
        "time", "hardware", "merkle_root"
    ]
    
    missing = []
    for field in required_fields:
        if field not in result:
            missing.append(field)
    
    return missing