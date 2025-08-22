"""Integration of leakage tracking with POT verification system."""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pot.core.challenge import ChallengeConfig, generate_challenges
from pot.core.sequential import sequential_verify
from pot.security.leakage import ReusePolicy, LeakageAuditor, compute_challenge_hash
from pot.audit import generate_session_id, write_audit_record, generate_nonce


class SecureVerificationProtocol:
    """
    Secure POT verification protocol with leakage tracking and audit trail.
    
    Integrates:
    - Challenge generation with PRF
    - Sequential verification with confidence sequences
    - Leakage tracking and reuse policy enforcement
    - Audit trail with commit-reveal
    """
    
    def __init__(
        self,
        master_key: bytes,
        u: int = 5,                    # Max uses per challenge
        rho_max: float = 0.3,           # Max leakage ratio
        alpha: float = 0.05,            # Type I error rate
        beta: float = 0.05,             # Type II error rate
        tau: float = 0.1,               # Decision threshold
        n_max: int = 1000,              # Max samples per verification
        persistence_dir: Optional[str] = None
    ):
        """
        Initialize secure verification protocol.
        
        Args:
            master_key: Master key for challenge generation
            u: Maximum uses per challenge
            rho_max: Maximum allowed leakage ratio
            alpha: False acceptance rate
            beta: False rejection rate
            tau: Decision threshold for verification
            n_max: Maximum samples before forced decision
            persistence_dir: Directory for persistent state
        """
        self.master_key = master_key
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.n_max = n_max
        
        # Set up persistence paths
        if persistence_dir:
            persist_path = Path(persistence_dir)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            reuse_path = persist_path / "reuse_state.json"
            audit_path = persist_path / "audit_log.jsonl"
        else:
            reuse_path = None
            audit_path = None
        
        # Initialize components
        self.reuse_policy = ReusePolicy(u, rho_max, persistence_path=str(reuse_path) if reuse_path else None)
        self.auditor = LeakageAuditor(log_path=str(audit_path) if audit_path else None)
        
        # Challenge library
        self.challenge_library: Dict[str, Dict[str, Any]] = {}
    
    def generate_challenge_set(
        self,
        n: int,
        family: str,
        params: Dict[str, Any],
        session_nonce: bytes
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Generate a set of challenges with leakage control.
        
        Args:
            n: Number of challenges to generate
            family: Challenge family
            params: Family-specific parameters
            session_nonce: Session-specific nonce
        
        Returns:
            Tuple of (challenge_ids, challenge_data)
        """
        # Configure challenge generation
        config = ChallengeConfig(
            master_key_hex=self.master_key.hex(),
            session_nonce_hex=session_nonce.hex(),
            n=n * 2,  # Generate extra to account for reuse limits
            family=family,
            params=params
        )
        
        # Generate challenges
        result = generate_challenges(config)
        all_challenges = result["items"]
        
        # Filter based on reuse policy
        available_challenges = []
        challenge_ids = []
        
        for challenge in all_challenges:
            # Compute stable ID for challenge
            cid = compute_challenge_hash(challenge)
            
            # Check if we can use this challenge (without recording yet)
            if cid not in self.reuse_policy.usage or self.reuse_policy.usage[cid].use_count < self.reuse_policy.u:
                available_challenges.append(challenge)
                challenge_ids.append(cid)
                
                # Store in library
                self.challenge_library[cid] = {
                    "data": challenge,
                    "family": family,
                    "created": time.time()
                }
                
                if len(available_challenges) >= n:
                    break
        
        if len(available_challenges) < n:
            raise ValueError(f"Could only generate {len(available_challenges)} usable challenges, needed {n}")
        
        return challenge_ids[:n], available_challenges[:n]
    
    def verify_model(
        self,
        model_evaluator,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform secure model verification with leakage tracking.
        
        Args:
            model_evaluator: Function that takes challenge and returns similarity score
            session_id: Optional session ID (generated if not provided)
        
        Returns:
            Verification result dictionary
        """
        if session_id is None:
            session_id = generate_session_id()
        
        # Start session tracking
        self.reuse_policy.start_session(session_id)
        start_time = time.time()
        
        # Generate session nonce
        session_nonce = generate_nonce()
        
        # Generate challenges with leakage control
        # Start with a smaller set to avoid exhausting challenges
        initial_n = min(100, self.n_max)
        try:
            challenge_ids, challenges = self.generate_challenge_set(
                n=initial_n,
                family="vision:freq",  # Example family
                params={"freq_range": [0.1, 10.0], "contrast_range": [0.1, 1.0]},
                session_nonce=session_nonce
            )
        except ValueError as e:
            # Too many challenges exhausted
            self.auditor.log_policy_violation(
                session_id,
                "insufficient_fresh_challenges",
                {"error": str(e)}
            )
            return {
                "error": "Insufficient fresh challenges available",
                "session_id": session_id,
                "decision": "reject_id"
            }
        
        # Check leakage threshold before proceeding
        is_safe, observed_rho = self.reuse_policy.check_leakage_threshold(challenge_ids)
        
        if not is_safe:
            self.auditor.log_policy_violation(
                session_id,
                "leakage_threshold_exceeded",
                {"observed_rho": observed_rho, "max_rho": self.reuse_policy.rho_max}
            )
            return {
                "error": "Leakage threshold exceeded",
                "session_id": session_id,
                "observed_rho": observed_rho,
                "decision": "reject_id"
            }
        
        # Create similarity score stream and record challenge uses
        def similarity_stream():
            for i, challenge in enumerate(challenges):
                # Record the actual use now
                cid = challenge_ids[i]
                self.reuse_policy.record_use(cid, session_id)
                
                # Evaluate model on challenge
                score = model_evaluator(challenge)
                yield score
        
        # Perform sequential verification
        decision, trail = sequential_verify(
            stream=similarity_stream(),
            tau=self.tau,
            alpha=self.alpha,
            beta=self.beta,
            n_max=len(challenges)
        )
        
        # End session and get stats
        session_stats = self.reuse_policy.end_session(session_id)
        
        # Log session completion
        if session_stats:
            self.auditor.log_session_complete(session_id, session_stats)
        
        # Create verification result
        result = {
            "session_id": session_id,
            "decision": decision["type"],
            "stopping_time": decision["stopping_time"],
            "final_mean": decision["final_mean"],
            "confidence_interval": decision.get("confidence_interval"),
            "leakage_stats": {
                "observed_rho": session_stats.observed_rho if session_stats else 0.0,
                "total_challenges": session_stats.total_challenges if session_stats else 0,
                "leaked_challenges": session_stats.leaked_challenges if session_stats else 0
            },
            "duration_seconds": time.time() - start_time,
            "trail_length": len(trail)
        }
        
        # Write audit record
        write_audit_record(
            session_id=session_id,
            model_id="test_model",  # Would be provided by caller
            family="vision:freq",
            alpha=self.alpha,
            beta=self.beta,
            boundary=self.tau,
            nonce=session_nonce,
            commitment=b"placeholder",  # Would compute actual commitment
            prf_info={
                "algorithm": "HMAC-SHA256",
                "key_derivation": "PRF",
                "seed_length": 32
            },
            reuse_policy=f"u={self.reuse_policy.u}, rho_max={self.reuse_policy.rho_max}",
            env={
                "python_version": sys.version,
                "platform": sys.platform
            },
            artifacts=result
        )
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status including leakage statistics."""
        usage_stats = self.reuse_policy.get_usage_stats()
        audit_report = self.auditor.generate_report()
        
        return {
            "usage_statistics": usage_stats,
            "audit_summary": audit_report,
            "challenge_library_size": len(self.challenge_library),
            "current_session": self.reuse_policy.current_session
        }


def demo_integrated_verification():
    """Demonstrate integrated verification with leakage tracking."""
    print("Integrated POT Verification Demo")
    print("=" * 50)
    
    import tempfile
    import numpy as np
    
    # Set up protocol
    master_key = b"test_master_key_32_bytes_long!!!"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        protocol = SecureVerificationProtocol(
            master_key=master_key,
            u=3,  # Max 3 uses per challenge
            rho_max=0.3,  # Max 30% leakage
            persistence_dir=tmpdir
        )
        
        # Define mock model evaluators
        def genuine_model(challenge):
            """Genuine model: low similarity scores."""
            return np.random.beta(2, 20)  # Mean ~0.09
        
        def adversarial_model(challenge):
            """Adversarial model: high similarity scores."""
            return np.random.beta(20, 2)  # Mean ~0.91
        
        # Test 1: Verify genuine model
        print("\nTest 1: Genuine Model Verification")
        print("-" * 40)
        result1 = protocol.verify_model(genuine_model, "genuine_test")
        print(f"Decision: {result1.get('decision', 'error')}")
        if 'error' in result1:
            print(f"Error: {result1['error']}")
        else:
            print(f"Stopping time: {result1['stopping_time']}")
            print(f"Final mean: {result1['final_mean']:.4f}")
            print(f"Leakage: {result1['leakage_stats']['observed_rho']:.2%}")
        
        # Test 2: Verify adversarial model
        print("\nTest 2: Adversarial Model Verification")
        print("-" * 40)
        result2 = protocol.verify_model(adversarial_model, "adversarial_test")
        print(f"Decision: {result2.get('decision', 'error')}")
        if 'error' in result2:
            print(f"Error: {result2['error']}")
        else:
            print(f"Stopping time: {result2.get('stopping_time', 'N/A')}")
            print(f"Final mean: {result2.get('final_mean', 0):.4f}")
            print(f"Leakage: {result2.get('leakage_stats', {}).get('observed_rho', 0):.2%}")
        
        # Test 3: Try verification with reused challenges
        print("\nTest 3: Verification with Reuse")
        print("-" * 40)
        result3 = protocol.verify_model(genuine_model, "reuse_test")
        print(f"Decision: {result3['decision']}")
        if "error" in result3:
            print(f"Error: {result3['error']}")
        else:
            print(f"Leakage: {result3['leakage_stats']['observed_rho']:.2%}")
        
        # Show system status
        print("\nSystem Status")
        print("-" * 40)
        status = protocol.get_system_status()
        print(f"Total challenges used: {status['usage_statistics']['total_challenges']}")
        print(f"Reused challenges: {status['usage_statistics']['reused_challenges']}")
        print(f"Exhausted challenges: {status['usage_statistics']['exhausted_challenges']}")
        print(f"Total verifications: {status['audit_summary']['unique_sessions']}")
        print(f"Policy violations: {status['audit_summary']['policy_violations']}")


if __name__ == "__main__":
    demo_integrated_verification()