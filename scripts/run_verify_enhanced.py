#!/usr/bin/env python
"""
Enhanced verification CLI with new protocol features.

Supports:
- Sequential verification with confidence sequences
- PRF-based challenge generation
- Leakage tracking and reuse policies
- Commit-reveal protocol
- Comprehensive artifact generation
- Visualization generation
"""

import argparse
import json
import yaml
import numpy as np
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Allow running as script from project root
sys.path.append(str(Path(__file__).parent.parent))

# Core imports
from pot.core.canonicalize import canonicalize_logits
from pot.core.challenge import ChallengeConfig, generate_challenges
from pot.core.governance import new_session_nonce
from pot.core.sequential import sequential_verify
from pot.core.boundaries import CSState, eb_radius as compute_eb_radius
from pot.core.prf import prf_derive_key
from pot.core.stats import far_frr, t_statistic
from pot.core.logging import StructuredLogger

# Security imports
from pot.security.leakage import ReusePolicy, LeakageAuditor, compute_challenge_hash
from pot.audit import (
    generate_session_id,
    write_audit_record,
    serialize_for_commit,
    make_commitment,
    verify_commitment,
    generate_nonce
)

# Evaluation imports
from pot.eval.metrics import dist_logits_l2, dist_kl

# Model loading
from pot.models import load_model

# Plotting imports
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, det_curve
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced POT verification with new protocol features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  %(prog)s --config config.yaml
  
  # Strict verification with tight bounds
  %(prog)s --config config.yaml --alpha 0.001 --beta 0.001 --tau-id 0.01
  
  # With PRF and challenge control
  %(prog)s --config config.yaml --master-key <hex> --family vision:freq \\
           --params '{"freq_range": [0.1, 10.0]}' --reuse-u 5 --rho-max 0.3
  
Output artifacts:
  commit.json       - Private pre-response commitment data
  commitment.json   - Public commitment hash  
  reveal.json       - Post-response reveal data
  decision.json     - Verification decision and statistics
  audit_*.json      - Complete audit record
  plots/           - Visualization plots
        """
    )
    
    # Required arguments
    parser.add_argument("--config", required=True, help="Config YAML file")
    
    # Error bounds
    parser.add_argument("--alpha", type=float, default=0.01,
                       help="Type I error bound (false acceptance rate)")
    parser.add_argument("--beta", type=float, default=0.01,
                       help="Type II error bound (false rejection rate)")
    
    # Decision parameters
    parser.add_argument("--tau-id", type=float, default=0.05,
                       help="Identity threshold for verification")
    parser.add_argument("--n-max", type=int, default=512,
                       help="Maximum challenges before forced decision")
    parser.add_argument("--boundary", choices=["EB", "MB"], default="EB",
                       help="Boundary type (Empirical Bernstein or Mixture)")
    
    # PRF and challenge generation
    parser.add_argument("--master-key", type=str, default=None,
                       help="Master key for PRF (hex string)")
    parser.add_argument("--nonce", type=str, default=None,
                       help="Session nonce (hex string or auto-generate)")
    parser.add_argument("--family", type=str, default="vision:freq",
                       help="Challenge family name")
    parser.add_argument("--params", type=str, default=None,
                       help="Challenge parameters as JSON string")
    
    # Equivalence and transformations
    parser.add_argument("--equiv", nargs="+", default=[],
                       help="List of allowed equivalence transforms")
    parser.add_argument("--wrapper-budget-proxy", type=float, default=0.1,
                       help="Maximum proxy fraction for wrapper detection")
    
    # Leakage control
    parser.add_argument("--reuse-u", type=int, default=5,
                       help="Maximum uses per challenge")
    parser.add_argument("--rho-max", type=float, default=0.3,
                       help="Maximum leakage ratio")
    
    # Output
    parser.add_argument("--outdir", type=str, default="outputs/verify",
                       help="Output directory for artifacts")
    
    # Additional options
    parser.add_argument("--cpu-only", action="store_true",
                       help="Run on CPU only")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    
    return parser.parse_args()


def setup_environment(args):
    """Set up environment based on arguments."""
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (outdir / "plots").mkdir(exist_ok=True)
    
    return outdir


def load_configuration(args) -> Dict[str, Any]:
    """Load and merge configuration from file and command-line."""
    # Load base config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override with command-line arguments
    config["verification"] = config.get("verification", {})
    config["verification"]["alpha"] = args.alpha
    config["verification"]["beta"] = args.beta
    config["verification"]["tau_id"] = args.tau_id
    config["verification"]["n_max"] = args.n_max
    config["verification"]["boundary_type"] = args.boundary
    
    # Challenge configuration
    if args.params:
        challenge_params = json.loads(args.params)
    else:
        # Try to get from config file
        challenge_params = None
        for fam in config.get("challenges", {}).get("families", []):
            if fam["family"] == args.family:
                challenge_params = fam["params"]
                break
        
        if challenge_params is None:
            # Default parameters
            challenge_params = {
                "freq_range": [0.1, 10.0],
                "contrast_range": [0.1, 1.0]
            }
    
    config["challenge_params"] = challenge_params
    config["challenge_family"] = args.family
    
    # Leakage configuration
    config["leakage"] = {
        "reuse_u": args.reuse_u,
        "rho_max": args.rho_max
    }
    
    return config


def generate_verification_challenges(
    master_key: bytes,
    nonce: bytes,
    family: str,
    params: Dict[str, Any],
    n: int,
    reuse_policy: Optional[ReusePolicy] = None
) -> Tuple[List[str], List[Any], Dict[str, Any]]:
    """
    Generate challenges with leakage control.
    
    Returns:
        Tuple of (challenge_ids, challenges, metadata)
    """
    # Configure challenge generation
    cfg = ChallengeConfig(
        master_key_hex=master_key.hex(),
        session_nonce_hex=nonce.hex(),
        n=n * 2,  # Generate extra to account for reuse
        family=family,
        params=params
    )
    
    # Generate challenges
    result = generate_challenges(cfg)
    all_challenges = result["items"]
    
    # Filter based on reuse policy if provided
    if reuse_policy:
        available_challenges = []
        challenge_ids = []
        
        for challenge in all_challenges:
            cid = compute_challenge_hash(challenge)
            
            # Check if we can use this challenge
            if cid not in reuse_policy.usage or \
               reuse_policy.usage[cid].use_count < reuse_policy.u:
                available_challenges.append(challenge)
                challenge_ids.append(cid)
                
                if len(available_challenges) >= n:
                    break
    else:
        # No reuse policy, use all challenges
        available_challenges = all_challenges[:n]
        challenge_ids = [compute_challenge_hash(c) for c in available_challenges]
    
    metadata = {
        "challenge_id": result["challenge_id"],
        "family": family,
        "salt": result["salt"],
        "total_generated": len(all_challenges),
        "used": len(available_challenges)
    }
    
    return challenge_ids, available_challenges, metadata


def run_sequential_verification(
    model_evaluator,
    challenges: List[Any],
    challenge_ids: List[str],
    tau: float,
    alpha: float,
    beta: float,
    n_max: int,
    reuse_policy: Optional[ReusePolicy] = None,
    session_id: Optional[str] = None
) -> Tuple[Dict[str, Any], List[Tuple]]:
    """
    Run sequential verification protocol.
    
    Returns:
        Tuple of (decision, trail)
    """
    # Create distance stream
    def distance_stream():
        for i, challenge in enumerate(challenges[:n_max]):
            # Record challenge use if policy exists
            if reuse_policy and session_id:
                reuse_policy.record_use(challenge_ids[i], session_id)
            
            # Evaluate model
            distance = model_evaluator(challenge)
            yield distance
    
    # Run sequential verification
    decision, trail = sequential_verify(
        stream=distance_stream(),
        tau=tau,
        alpha=alpha,
        beta=beta,
        n_max=min(n_max, len(challenges))
    )
    
    return decision, trail


def generate_commit_reveal_artifacts(
    outdir: Path,
    session_id: str,
    master_key: bytes,
    nonce: bytes,
    challenges: List[Any],
    challenge_ids: List[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate commit-reveal protocol artifacts."""
    
    # Prepare data for commitment
    ranges = [(0, len(challenges))]
    context = {
        "session_id": session_id,
        "family": config["challenge_family"],
        "params": config["challenge_params"],
        "tau": config["verification"]["tau_id"],
        "alpha": config["verification"]["alpha"],
        "beta": config["verification"]["beta"]
    }
    
    # Serialize for commitment
    data_to_commit = serialize_for_commit(challenge_ids, ranges, context)
    
    # Create commitment
    commitment = make_commitment(master_key, nonce, data_to_commit)
    
    # Save private commit data
    commit_data = {
        "session_id": session_id,
        "timestamp": time.time(),
        "challenge_ids": challenge_ids,
        "ranges": ranges,
        "context": context,
        "nonce": nonce.hex()
    }
    
    with open(outdir / "commit.json", "w") as f:
        json.dump(commit_data, f, indent=2)
    
    # Save public commitment
    commitment_data = {
        "session_id": session_id,
        "commitment": commitment.hex(),
        "timestamp": time.time()
    }
    
    with open(outdir / "commitment.json", "w") as f:
        json.dump(commitment_data, f, indent=2)
    
    # Prepare reveal data (saved after verification)
    reveal_data = {
        "session_id": session_id,
        "challenge_ids": challenge_ids,
        "ranges": ranges,
        "context": context,
        "nonce": nonce.hex(),
        "commitment": commitment.hex()
    }
    
    return reveal_data


def generate_decision_artifact(
    outdir: Path,
    session_id: str,
    decision: Dict[str, Any],
    trail: List[Tuple],
    config: Dict[str, Any],
    leakage_stats: Optional[Dict[str, Any]] = None
) -> None:
    """Generate decision.json with specified schema."""
    
    # Extract key information
    stopping_time = decision["stopping_time"]
    final_mean = decision["final_mean"]
    
    # Get final radius (use alpha radius)
    final_radius = decision.get("final_radius_alpha", float("inf"))
    
    # Confidence interval
    ci = decision.get("confidence_interval", (0.0, 1.0))
    
    # Create decision artifact matching schema
    decision_artifact = {
        "session_id": session_id,
        "decision": decision["type"],  # accept_id, reject_id, or conservative_reject
        "stopping_time": stopping_time,
        "mean": final_mean,
        "radius": final_radius,
        "confidence_interval": {
            "lower": ci[0],
            "upper": ci[1]
        },
        "tau": config["verification"]["tau_id"],
        "alpha": config["verification"]["alpha"],
        "beta": config["verification"]["beta"],
        "n_max": config["verification"]["n_max"],
        "boundary_type": config["verification"]["boundary_type"],
        "timestamp": time.time()
    }
    
    # Add leakage statistics if available
    if leakage_stats:
        decision_artifact["leakage"] = leakage_stats
    
    # Add trail summary
    if trail:
        decision_artifact["trail_summary"] = {
            "length": len(trail),
            "initial_mean": trail[0][1] if trail else 0.0,
            "initial_radius": trail[0][2] if trail else float("inf"),
            "final_mean": trail[-1][1] if trail else 0.0,
            "final_radius": trail[-1][2] if trail else float("inf")
        }
    
    # Save decision artifact
    with open(outdir / "decision.json", "w") as f:
        json.dump(decision_artifact, f, indent=2)


def generate_plots(
    outdir: Path,
    trail: List[Tuple],
    config: Dict[str, Any]
) -> None:
    """Generate visualization plots."""
    if not HAS_PLOTTING:
        print("Plotting libraries not available, skipping plots")
        return
    
    plots_dir = outdir / "plots"
    
    # Extract trail data
    if trail:
        times = [t[0] for t in trail]
        means = [t[1] for t in trail]
        radii_alpha = [t[2] for t in trail]
        radii_beta = [t[3] for t in trail]
        
        # Plot 1: Stopping time visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Mean with confidence bounds
        ax1.plot(times, means, 'b-', label='Mean')
        lower_alpha = [m - r for m, r in zip(means, radii_alpha)]
        upper_alpha = [m + r for m, r in zip(means, radii_alpha)]
        ax1.fill_between(times, lower_alpha, upper_alpha, alpha=0.3, label=f'α={config["verification"]["alpha"]}')
        
        # Add threshold line
        tau = config["verification"]["tau_id"]
        ax1.axhline(y=tau, color='r', linestyle='--', label=f'τ={tau}')
        
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Mean Distance')
        ax1.set_title('Sequential Verification Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Radius evolution
        ax2.plot(times, radii_alpha, 'g-', label='Alpha radius')
        ax2.plot(times, radii_beta, 'r-', label='Beta radius')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Confidence Radius')
        ax2.set_title('Confidence Radius Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "stopping_time.png", dpi=150)
        plt.close()
        
        # Plot 2: ROC curve (if we have enough data)
        if len(means) > 10:
            # Simulate ROC data based on trail
            # This is a simplified visualization
            thresholds = np.linspace(0, 1, 100)
            tpr = []
            fpr = []
            
            for thresh in thresholds:
                # Estimate TPR/FPR based on mean and variance
                final_mean = means[-1]
                final_std = np.std(means) if len(means) > 1 else 0.1
                
                # Simple normal approximation
                from scipy.stats import norm
                tpr.append(1 - norm.cdf(thresh, final_mean, final_std))
                fpr.append(1 - norm.cdf(thresh, 0, final_std))
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(fpr, tpr, 'b-', linewidth=2)
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve (Estimated)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "roc.png", dpi=150)
            plt.close()
        
        print(f"Plots saved to {plots_dir}/")


def main():
    """Main verification workflow."""
    args = parse_args()
    
    # Set up environment
    outdir = setup_environment(args)
    
    # Load configuration
    config = load_configuration(args)
    
    # Initialize logging
    logger = StructuredLogger(str(outdir))
    
    # Generate session ID
    session_id = generate_session_id()
    
    if args.verbose:
        print(f"Session ID: {session_id}")
        print(f"Output directory: {outdir}")
    
    # Set up master key and nonce
    if args.master_key:
        master_key = bytes.fromhex(args.master_key)
    else:
        master_key = os.urandom(32)
    
    if args.nonce:
        if len(args.nonce) == 64:  # Hex string
            nonce = bytes.fromhex(args.nonce)
        else:
            # Use as seed for deterministic nonce
            nonce = hashlib.sha256(args.nonce.encode()).digest()
    else:
        nonce = os.urandom(32)
    
    # Initialize leakage tracking
    reuse_policy = ReusePolicy(
        u=args.reuse_u,
        rho_max=args.rho_max,
        persistence_path=str(outdir / "reuse_state.json")
    )
    
    auditor = LeakageAuditor(
        log_path=str(outdir / "audit_log.jsonl")
    )
    
    # Start session
    reuse_policy.start_session(session_id)
    
    # Load models
    ref_model = load_model(config["models"]["reference_path"], cpu_only=args.cpu_only).eval()
    test_model = load_model(config["models"]["test_path"], cpu_only=args.cpu_only).eval()
    
    # Generate challenges
    if args.verbose:
        print("Generating challenges...")
    
    challenge_ids, challenges, challenge_metadata = generate_verification_challenges(
        master_key=master_key,
        nonce=nonce,
        family=args.family,
        params=config["challenge_params"],
        n=args.n_max,
        reuse_policy=reuse_policy
    )
    
    # Check leakage threshold
    is_safe, observed_rho = reuse_policy.check_leakage_threshold(challenge_ids)
    
    if not is_safe:
        print(f"Warning: Leakage threshold exceeded (ρ={observed_rho:.2f} > {args.rho_max})")
        auditor.log_policy_violation(
            session_id,
            "leakage_threshold_exceeded",
            {"observed_rho": observed_rho, "max_rho": args.rho_max}
        )
    
    # Generate commit artifacts
    reveal_data = generate_commit_reveal_artifacts(
        outdir=outdir,
        session_id=session_id,
        master_key=master_key,
        nonce=nonce,
        challenges=challenges,
        challenge_ids=challenge_ids,
        config=config
    )
    
    # Define model evaluator
    import torch
    
    def model_evaluator(challenge):
        """Evaluate model on challenge and return distance."""
        # Convert challenge to tensor (adjust based on actual format)
        if isinstance(challenge, dict):
            vals = [challenge[k] for k in sorted(challenge.keys())]
            x = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
        else:
            x = torch.tensor(challenge, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits_ref = ref_model(x).squeeze(0).numpy()
            logits_test = test_model(x).squeeze(0).numpy()
        
        # Canonicalize
        logits_ref = canonicalize_logits(logits_ref)
        logits_test = canonicalize_logits(logits_test)
        
        # Compute distance (using L2 as default)
        distance = dist_logits_l2(logits_ref, logits_test)
        
        return distance
    
    # Run sequential verification
    if args.verbose:
        print("Running sequential verification...")
    
    decision, trail = run_sequential_verification(
        model_evaluator=model_evaluator,
        challenges=challenges,
        challenge_ids=challenge_ids,
        tau=args.tau_id,
        alpha=args.alpha,
        beta=args.beta,
        n_max=args.n_max,
        reuse_policy=reuse_policy,
        session_id=session_id
    )
    
    # End session and get leakage stats
    session_stats = reuse_policy.end_session(session_id)
    
    if session_stats:
        leakage_stats = {
            "observed_rho": session_stats.observed_rho,
            "total_challenges": session_stats.total_challenges,
            "leaked_challenges": session_stats.leaked_challenges
        }
        auditor.log_session_complete(session_id, session_stats)
    else:
        leakage_stats = None
    
    # Save reveal data
    with open(outdir / "reveal.json", "w") as f:
        json.dump(reveal_data, f, indent=2)
    
    # Generate decision artifact
    generate_decision_artifact(
        outdir=outdir,
        session_id=session_id,
        decision=decision,
        trail=trail,
        config=config,
        leakage_stats=leakage_stats
    )
    
    # Generate audit record
    write_audit_record(
        session_id=session_id,
        model_id=config["models"]["test_path"],
        family=args.family,
        alpha=args.alpha,
        beta=args.beta,
        boundary=args.tau_id,
        nonce=nonce,
        commitment=bytes.fromhex(reveal_data["commitment"]),
        prf_info={
            "algorithm": "HMAC-SHA256",
            "key_derivation": "PRF",
            "seed_length": len(master_key)
        },
        reuse_policy=f"u={args.reuse_u}, rho_max={args.rho_max}",
        env={
            "python_version": sys.version,
            "platform": sys.platform,
            "wrapper_budget": args.wrapper_budget_proxy,
            "equiv_transforms": args.equiv
        },
        artifacts={
            "decision": decision["type"],
            "stopping_time": decision["stopping_time"],
            "challenge_metadata": challenge_metadata
        },
        output_dir=str(outdir)
    )
    
    # Generate plots
    if not args.no_plots:
        generate_plots(outdir, trail, config)
    
    # Print summary
    print(f"\nVerification Complete")
    print(f"  Session ID: {session_id}")
    print(f"  Decision: {decision['type']}")
    print(f"  Stopping time: {decision['stopping_time']}")
    print(f"  Final mean: {decision['final_mean']:.4f}")
    print(f"  Confidence interval: [{decision.get('confidence_interval', [0,1])[0]:.4f}, "
          f"{decision.get('confidence_interval', [0,1])[1]:.4f}]")
    
    if leakage_stats:
        print(f"  Leakage ratio: {leakage_stats['observed_rho']:.2%}")
    
    print(f"\nArtifacts saved to: {outdir}/")
    print(f"  - commit.json: Private pre-response data")
    print(f"  - commitment.json: Public commitment hash")
    print(f"  - reveal.json: Post-response reveal data")
    print(f"  - decision.json: Decision and statistics")
    print(f"  - audit_*.json: Complete audit record")
    
    if not args.no_plots and HAS_PLOTTING:
        print(f"  - plots/: Visualizations")
    
    # Return decision for scripting
    return decision["type"]


if __name__ == "__main__":
    decision = main()
    sys.exit(0 if decision == "accept_id" else 1)