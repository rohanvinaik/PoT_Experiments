"""Leakage tracking and challenge reuse policy enforcement for POT."""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple, Any
import hashlib


@dataclass
class ChallengeUsage:
    """Track usage of a single challenge."""
    challenge_id: str
    use_count: int = 0
    first_used: Optional[float] = None
    last_used: Optional[float] = None
    sessions: List[str] = field(default_factory=list)
    
    def record_use(self, session_id: str) -> None:
        """Record a use of this challenge."""
        now = time.time()
        self.use_count += 1
        if self.first_used is None:
            self.first_used = now
        self.last_used = now
        if session_id not in self.sessions:
            self.sessions.append(session_id)


@dataclass
class LeakageStats:
    """Statistics about leakage in a session."""
    session_id: str
    total_challenges: int = 0
    leaked_challenges: int = 0
    observed_rho: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def update(self, leaked: int, total: int) -> None:
        """Update leakage statistics."""
        self.leaked_challenges = leaked
        self.total_challenges = total
        self.observed_rho = leaked / total if total > 0 else 0.0


class ReusePolicy:
    """
    Challenge reuse policy enforcement with leakage tracking.
    
    Tracks challenge usage across sessions and enforces reuse limits
    to prevent adversaries from learning too much about the challenge set.
    """
    
    def __init__(
        self,
        u: int,
        rho_max: float,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize reuse policy.
        
        Args:
            u: Maximum number of uses per challenge
            rho_max: Maximum allowed leakage ratio (fraction of challenges that can be reused)
            persistence_path: Optional path to persist usage data
        """
        self.u = u
        self.rho_max = rho_max
        self.persistence_path = persistence_path
        
        # Challenge usage tracking
        self.usage: Dict[str, ChallengeUsage] = {}
        
        # Session tracking
        self.sessions: Dict[str, LeakageStats] = {}
        self.current_session: Optional[str] = None
        
        # Load persisted data if available
        if persistence_path:
            self.load_state()
    
    def record_use(self, challenge_id: str, session_id: Optional[str] = None) -> bool:
        """
        Record use of a challenge and check if it's under the limit.
        
        Args:
            challenge_id: Unique identifier for the challenge
            session_id: Optional session identifier
        
        Returns:
            True if challenge can be used (under limit), False otherwise
        """
        # Ensure challenge_id is a string
        if not isinstance(challenge_id, str):
            challenge_id = str(challenge_id)
        
        # Get or create usage record
        if challenge_id not in self.usage:
            self.usage[challenge_id] = ChallengeUsage(challenge_id)
        
        usage = self.usage[challenge_id]
        
        # Check if under limit
        if usage.use_count >= self.u:
            return False  # Challenge has been used too many times
        
        # Record the use
        if session_id is None:
            session_id = self.current_session or "default"
        
        usage.record_use(session_id)
        
        # Update session stats if tracking
        if session_id in self.sessions:
            self.sessions[session_id].total_challenges += 1
        
        # Persist if configured
        if self.persistence_path:
            self.save_state()
        
        return True
    
    def observed_rho(self, leaked_pairs: int, n: int) -> float:
        """
        Compute observed leakage ratio.
        
        Args:
            leaked_pairs: Number of challenge-response pairs that have been leaked/reused
            n: Total number of challenges in the session
        
        Returns:
            Leakage ratio rho = leaked_pairs / n
        """
        if n <= 0:
            return 0.0
        return leaked_pairs / n
    
    def get_leaked_count(self, challenge_ids: List[str]) -> int:
        """
        Count how many of the given challenges have been previously used.
        
        Args:
            challenge_ids: List of challenge IDs to check
        
        Returns:
            Number of challenges that have been used before
        """
        leaked_count = 0
        for cid in challenge_ids:
            cid_str = str(cid)
            if cid_str in self.usage and self.usage[cid_str].use_count > 0:
                leaked_count += 1
        return leaked_count
    
    def check_leakage_threshold(self, challenge_ids: List[str]) -> Tuple[bool, float]:
        """
        Check if using these challenges would exceed leakage threshold.
        
        Args:
            challenge_ids: List of challenge IDs to check
        
        Returns:
            Tuple of (is_safe, observed_rho) where is_safe indicates if
            leakage is below threshold
        """
        leaked = self.get_leaked_count(challenge_ids)
        n = len(challenge_ids)
        rho = self.observed_rho(leaked, n)
        
        return rho <= self.rho_max, rho
    
    def start_session(self, session_id: str) -> None:
        """Start tracking a new verification session."""
        self.current_session = session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = LeakageStats(session_id)
    
    def end_session(self, session_id: Optional[str] = None) -> Optional[LeakageStats]:
        """
        End a verification session and return its statistics.
        
        Args:
            session_id: Session to end (or current session if None)
        
        Returns:
            Session statistics if found
        """
        if session_id is None:
            session_id = self.current_session
        
        if session_id and session_id in self.sessions:
            stats = self.sessions[session_id]
            # Calculate final leakage
            leaked = sum(
                1 for usage in self.usage.values()
                if session_id in usage.sessions and usage.use_count > 1
            )
            total = sum(
                1 for usage in self.usage.values()
                if session_id in usage.sessions
            )
            stats.update(leaked, total)
            
            if self.persistence_path:
                self.save_state()
            
            return stats
        
        return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get overall usage statistics."""
        total_challenges = len(self.usage)
        reused_challenges = sum(1 for u in self.usage.values() if u.use_count > 1)
        exhausted_challenges = sum(1 for u in self.usage.values() if u.use_count >= self.u)
        
        use_distribution = defaultdict(int)
        for usage in self.usage.values():
            use_distribution[usage.use_count] += 1
        
        return {
            "total_challenges": total_challenges,
            "reused_challenges": reused_challenges,
            "exhausted_challenges": exhausted_challenges,
            "reuse_ratio": reused_challenges / total_challenges if total_challenges > 0 else 0,
            "exhaustion_ratio": exhausted_challenges / total_challenges if total_challenges > 0 else 0,
            "use_distribution": dict(use_distribution),
            "max_uses_allowed": self.u,
            "max_leakage_allowed": self.rho_max
        }
    
    def reset_challenge(self, challenge_id: str) -> None:
        """Reset usage counter for a specific challenge (admin function)."""
        if challenge_id in self.usage:
            del self.usage[challenge_id]
    
    def reset_all(self) -> None:
        """Reset all usage counters (admin function)."""
        self.usage.clear()
        self.sessions.clear()
        self.current_session = None
        
        if self.persistence_path:
            self.save_state()
    
    def save_state(self) -> None:
        """Persist usage data to disk."""
        if not self.persistence_path:
            return
        
        path = Path(self.persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "u": self.u,
            "rho_max": self.rho_max,
            "usage": {
                cid: {
                    "challenge_id": usage.challenge_id,
                    "use_count": usage.use_count,
                    "first_used": usage.first_used,
                    "last_used": usage.last_used,
                    "sessions": usage.sessions
                }
                for cid, usage in self.usage.items()
            },
            "sessions": {
                sid: asdict(stats)
                for sid, stats in self.sessions.items()
            },
            "current_session": self.current_session,
            "timestamp": time.time()
        }
        
        # Write atomically with temp file
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(state, f, indent=2)
        temp_path.replace(path)
    
    def load_state(self) -> bool:
        """
        Load persisted usage data from disk.
        
        Returns:
            True if state was loaded successfully
        """
        if not self.persistence_path:
            return False
        
        path = Path(self.persistence_path)
        if not path.exists():
            return False
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            # Restore usage data
            self.usage.clear()
            for cid, data in state.get("usage", {}).items():
                usage = ChallengeUsage(
                    challenge_id=data["challenge_id"],
                    use_count=data["use_count"],
                    first_used=data.get("first_used"),
                    last_used=data.get("last_used"),
                    sessions=data.get("sessions", [])
                )
                self.usage[cid] = usage
            
            # Restore session data
            self.sessions.clear()
            for sid, data in state.get("sessions", {}).items():
                stats = LeakageStats(
                    session_id=data["session_id"],
                    total_challenges=data.get("total_challenges", 0),
                    leaked_challenges=data.get("leaked_challenges", 0),
                    observed_rho=data.get("observed_rho", 0.0),
                    timestamp=data.get("timestamp", time.time())
                )
                self.sessions[sid] = stats
            
            self.current_session = state.get("current_session")
            
            return True
            
        except Exception as e:
            print(f"Failed to load state from {path}: {e}")
            return False


class LeakageAuditor:
    """
    Auditor for tracking and reporting leakage across the system.
    """
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize leakage auditor.
        
        Args:
            log_path: Optional path for audit logs
        """
        self.log_path = log_path
        self.events: List[Dict[str, Any]] = []
    
    def log_challenge_use(
        self,
        challenge_id: str,
        session_id: str,
        use_count: int,
        allowed: bool,
        rho: float
    ) -> None:
        """Log a challenge use event."""
        event = {
            "type": "challenge_use",
            "challenge_id": challenge_id,
            "session_id": session_id,
            "use_count": use_count,
            "allowed": allowed,
            "observed_rho": rho,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.events.append(event)
        
        if self.log_path:
            self._write_log(event)
    
    def log_session_complete(
        self,
        session_id: str,
        stats: LeakageStats
    ) -> None:
        """Log session completion with leakage statistics."""
        event = {
            "type": "session_complete",
            "session_id": session_id,
            "total_challenges": stats.total_challenges,
            "leaked_challenges": stats.leaked_challenges,
            "observed_rho": stats.observed_rho,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.events.append(event)
        
        if self.log_path:
            self._write_log(event)
    
    def log_policy_violation(
        self,
        session_id: str,
        violation_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Log a policy violation."""
        event = {
            "type": "policy_violation",
            "session_id": session_id,
            "violation_type": violation_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.events.append(event)
        
        if self.log_path:
            self._write_log(event)
    
    def _write_log(self, event: Dict[str, Any]) -> None:
        """Write event to log file."""
        if not self.log_path:
            return
        
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate summary report of leakage."""
        if not self.events:
            return {"error": "No events to report"}
        
        total_uses = sum(1 for e in self.events if e["type"] == "challenge_use")
        rejected_uses = sum(
            1 for e in self.events 
            if e["type"] == "challenge_use" and not e["allowed"]
        )
        
        sessions = {
            e["session_id"] for e in self.events 
            if "session_id" in e
        }
        
        avg_rho = 0.0
        rho_events = [
            e for e in self.events 
            if "observed_rho" in e and e["observed_rho"] is not None
        ]
        if rho_events:
            avg_rho = sum(e["observed_rho"] for e in rho_events) / len(rho_events)
        
        violations = [
            e for e in self.events 
            if e["type"] == "policy_violation"
        ]
        
        return {
            "total_events": len(self.events),
            "total_challenge_uses": total_uses,
            "rejected_uses": rejected_uses,
            "rejection_rate": rejected_uses / total_uses if total_uses > 0 else 0,
            "unique_sessions": len(sessions),
            "average_leakage_ratio": avg_rho,
            "policy_violations": len(violations),
            "violation_details": violations
        }


def compute_challenge_hash(challenge_data: Any) -> str:
    """
    Compute a stable hash for challenge data.
    
    Args:
        challenge_data: Challenge data (dict, list, or string)
    
    Returns:
        Hex string hash of the challenge
    """
    if isinstance(challenge_data, dict):
        # Sort keys for stability
        data_str = json.dumps(challenge_data, sort_keys=True)
    elif isinstance(challenge_data, (list, tuple)):
        data_str = json.dumps(list(challenge_data))
    else:
        data_str = str(challenge_data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]