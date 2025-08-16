"""
Comprehensive unit tests for leakage tracking and challenge reuse.

Tests:
- Reuse counter enforcement
- Leakage ratio (rho) calculation
- Policy violation detection
- Session management
- Persistence and recovery
"""

import pytest
import json
import hashlib
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pot.security.leakage import (
    ReusePolicy,
    LeakageAuditor,
    compute_challenge_hash,
    ChallengeUsage,
    LeakageStats as SessionStats  # Using the correct class name
)


class TestChallengeHashing:
    """Test challenge hash computation."""
    
    def test_hash_deterministic(self):
        """Test that challenge hashing is deterministic."""
        challenge = {"param1": 1.0, "param2": "test", "param3": [1, 2, 3]}
        
        # Compute multiple times
        hash1 = compute_challenge_hash(challenge)
        hash2 = compute_challenge_hash(challenge)
        hash3 = compute_challenge_hash(challenge)
        
        # All should be identical
        assert hash1 == hash2 == hash3
        
        # Should be a string
        assert isinstance(hash1, str)
        
        # Should have reasonable length (hex digest)
        assert len(hash1) == 64  # SHA256 hex digest
    
    def test_hash_sensitivity(self):
        """Test that hash is sensitive to challenge changes."""
        base_challenge = {"param": 1.0, "data": [1, 2, 3]}
        base_hash = compute_challenge_hash(base_challenge)
        
        # Change parameter value
        challenge1 = {"param": 1.1, "data": [1, 2, 3]}
        assert compute_challenge_hash(challenge1) != base_hash
        
        # Change list element
        challenge2 = {"param": 1.0, "data": [1, 2, 4]}
        assert compute_challenge_hash(challenge2) != base_hash
        
        # Add parameter
        challenge3 = {"param": 1.0, "data": [1, 2, 3], "extra": 0}
        assert compute_challenge_hash(challenge3) != base_hash
    
    def test_hash_different_types(self):
        """Test hashing of different challenge types."""
        # Dict challenge
        dict_challenge = {"key": "value"}
        dict_hash = compute_challenge_hash(dict_challenge)
        assert isinstance(dict_hash, str)
        
        # List challenge
        list_challenge = [1, 2, 3, 4, 5]
        list_hash = compute_challenge_hash(list_challenge)
        assert isinstance(list_hash, str)
        
        # String challenge
        str_challenge = "test_challenge_string"
        str_hash = compute_challenge_hash(str_challenge)
        assert isinstance(str_hash, str)
        
        # All should be different
        assert dict_hash != list_hash != str_hash


class TestReusePolicy:
    """Test challenge reuse policy enforcement."""
    
    def test_basic_reuse_tracking(self):
        """Test basic reuse counter tracking."""
        policy = ReusePolicy(u=3, rho_max=0.5)
        
        challenge_id = "test_challenge_001"
        session1 = "session_001"
        session2 = "session_002"
        
        # First use
        policy.record_use(challenge_id, session1)
        assert challenge_id in policy.usage
        assert policy.usage[challenge_id].use_count == 1
        assert session1 in policy.usage[challenge_id].sessions
        
        # Second use (same session)
        policy.record_use(challenge_id, session1)
        assert policy.usage[challenge_id].use_count == 1  # Same session doesn't increment
        
        # Third use (different session)
        policy.record_use(challenge_id, session2)
        assert policy.usage[challenge_id].use_count == 2
        assert session2 in policy.usage[challenge_id].sessions
    
    def test_reuse_limit_enforcement(self):
        """Test that reuse limit u is enforced."""
        policy = ReusePolicy(u=2, rho_max=0.5)
        
        challenge_id = "test_challenge"
        
        # Use in first session
        policy.record_use(challenge_id, "session1")
        assert policy.usage[challenge_id].use_count == 1
        
        # Use in second session (at limit)
        policy.record_use(challenge_id, "session2")
        assert policy.usage[challenge_id].use_count == 2
        
        # Check if challenge can be used again
        # This should be checked before recording use
        can_use = challenge_id not in policy.usage or \
                  policy.usage[challenge_id].use_count < policy.u
        assert not can_use  # Should not be able to use
    
    def test_leakage_ratio_calculation(self):
        """Test leakage ratio (rho) calculation."""
        policy = ReusePolicy(u=3, rho_max=0.3)
        
        # Start a session
        session_id = "test_session"
        policy.start_session(session_id)
        
        # Use some challenges
        fresh_challenges = [f"fresh_{i}" for i in range(70)]
        reused_challenges = [f"reused_{i}" for i in range(30)]
        
        # Mark reused challenges as previously used
        for cid in reused_challenges:
            policy.usage[cid] = ChallengeUsage()
            policy.usage[cid].use_count = 1
            policy.usage[cid].sessions.add("previous_session")
        
        # Record all challenges for this session
        all_challenges = fresh_challenges + reused_challenges
        for cid in all_challenges:
            policy.record_use(cid, session_id)
        
        # Check leakage
        is_safe, observed_rho = policy.check_leakage_threshold(all_challenges)
        
        # Should have 30% leakage (30 out of 100)
        assert abs(observed_rho - 0.3) < 0.01
        assert is_safe  # At threshold, should be safe
        
        # Add one more reused challenge
        extra_reused = "extra_reused"
        policy.usage[extra_reused] = ChallengeUsage()
        policy.usage[extra_reused].use_count = 1
        policy.usage[extra_reused].sessions.add("previous_session")
        
        all_challenges_plus = all_challenges + [extra_reused]
        policy.record_use(extra_reused, session_id)
        
        is_safe2, observed_rho2 = policy.check_leakage_threshold(all_challenges_plus)
        assert observed_rho2 > 0.3
        assert not is_safe2  # Over threshold
    
    def test_session_management(self):
        """Test session lifecycle management."""
        policy = ReusePolicy(u=5, rho_max=0.3)
        
        session_id = "test_session_123"
        
        # Start session
        policy.start_session(session_id)
        assert session_id in policy.active_sessions
        assert policy.active_sessions[session_id].total_challenges == 0
        
        # Use challenges
        challenges = [f"challenge_{i}" for i in range(10)]
        for cid in challenges:
            policy.record_use(cid, session_id)
        
        # End session
        stats = policy.end_session(session_id)
        assert stats is not None
        assert stats.total_challenges == 10
        assert stats.leaked_challenges == 0  # All fresh
        assert stats.observed_rho == 0.0
        assert session_id not in policy.active_sessions
    
    def test_multiple_sessions(self):
        """Test handling multiple concurrent sessions."""
        policy = ReusePolicy(u=3, rho_max=0.5)
        
        # Start multiple sessions
        sessions = [f"session_{i}" for i in range(5)]
        for sid in sessions:
            policy.start_session(sid)
        
        # Each session uses some challenges
        for i, sid in enumerate(sessions):
            challenges = [f"challenge_{sid}_{j}" for j in range(10)]
            for cid in challenges:
                policy.record_use(cid, sid)
        
        # Verify all sessions tracked correctly
        for sid in sessions:
            assert sid in policy.active_sessions
            assert policy.active_sessions[sid].total_challenges == 10
        
        # End sessions and verify stats
        for sid in sessions:
            stats = policy.end_session(sid)
            assert stats.total_challenges == 10
    
    def test_persistence(self):
        """Test persistence and recovery of reuse state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = str(Path(tmpdir) / "reuse_state.json")
            
            # Create policy and add some usage
            policy1 = ReusePolicy(u=3, rho_max=0.3, persistence_path=persistence_path)
            
            # Record some usage
            for i in range(10):
                cid = f"challenge_{i}"
                policy1.record_use(cid, f"session_{i % 3}")
            
            # Save state
            policy1.save_state()
            
            # Create new policy and load state
            policy2 = ReusePolicy(u=3, rho_max=0.3, persistence_path=persistence_path)
            policy2.load_state()
            
            # Should have same usage data
            assert len(policy2.usage) == len(policy1.usage)
            for cid in policy1.usage:
                assert cid in policy2.usage
                assert policy2.usage[cid].use_count == policy1.usage[cid].use_count
                assert policy2.usage[cid].sessions == policy1.usage[cid].sessions


class TestLeakageAuditor:
    """Test leakage auditing functionality."""
    
    def test_audit_logging(self):
        """Test audit log generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "audit.jsonl")
            auditor = LeakageAuditor(log_path=log_path)
            
            # Log various events
            session_id = "test_session"
            
            # Log session start
            auditor.log_session_start(session_id, {"param": "value"})
            
            # Log challenge use
            auditor.log_challenge_use(session_id, "challenge_1", is_reused=False)
            auditor.log_challenge_use(session_id, "challenge_2", is_reused=True)
            
            # Log policy violation
            auditor.log_policy_violation(
                session_id,
                "leakage_exceeded",
                {"observed_rho": 0.35, "max_rho": 0.3}
            )
            
            # Log session complete
            stats = SessionStats()
            stats.total_challenges = 10
            stats.leaked_challenges = 3
            stats.observed_rho = 0.3
            auditor.log_session_complete(session_id, stats)
            
            # Read and verify log
            with open(log_path) as f:
                lines = f.readlines()
            
            assert len(lines) == 5  # 5 events logged
            
            # Parse and check events
            events = [json.loads(line) for line in lines]
            
            # Check session start
            assert events[0]["event_type"] == "session_start"
            assert events[0]["session_id"] == session_id
            
            # Check challenge uses
            assert events[1]["event_type"] == "challenge_use"
            assert events[1]["is_reused"] == False
            assert events[2]["event_type"] == "challenge_use"
            assert events[2]["is_reused"] == True
            
            # Check policy violation
            assert events[3]["event_type"] == "policy_violation"
            assert events[3]["violation_type"] == "leakage_exceeded"
            
            # Check session complete
            assert events[4]["event_type"] == "session_complete"
            assert events[4]["stats"]["total_challenges"] == 10
    
    def test_audit_timestamps(self):
        """Test that audit events have proper timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "audit.jsonl")
            auditor = LeakageAuditor(log_path=log_path)
            
            start_time = time.time()
            
            # Log events with small delays
            auditor.log_session_start("session1", {})
            time.sleep(0.01)
            auditor.log_challenge_use("session1", "chal1", False)
            time.sleep(0.01)
            auditor.log_session_complete("session1", SessionStats())
            
            end_time = time.time()
            
            # Read events
            with open(log_path) as f:
                events = [json.loads(line) for line in f]
            
            # All should have timestamps
            for event in events:
                assert "timestamp" in event
                assert start_time <= event["timestamp"] <= end_time
            
            # Timestamps should be increasing
            for i in range(1, len(events)):
                assert events[i]["timestamp"] >= events[i-1]["timestamp"]
    
    def test_audit_persistence(self):
        """Test that audit logs persist across auditor instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "audit.jsonl")
            
            # First auditor
            auditor1 = LeakageAuditor(log_path=log_path)
            auditor1.log_session_start("session1", {})
            del auditor1
            
            # Second auditor (appends)
            auditor2 = LeakageAuditor(log_path=log_path)
            auditor2.log_session_start("session2", {})
            del auditor2
            
            # Read all events
            with open(log_path) as f:
                events = [json.loads(line) for line in f]
            
            assert len(events) == 2
            assert events[0]["session_id"] == "session1"
            assert events[1]["session_id"] == "session2"


class TestIntegration:
    """Integration tests for complete leakage tracking flow."""
    
    def test_complete_verification_flow(self):
        """Test complete flow with policy and auditor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            persistence_path = str(Path(tmpdir) / "reuse_state.json")
            audit_path = str(Path(tmpdir) / "audit.jsonl")
            
            policy = ReusePolicy(u=2, rho_max=0.25, persistence_path=persistence_path)
            auditor = LeakageAuditor(log_path=audit_path)
            
            # Session 1: Use some challenges
            session1 = "session_001"
            policy.start_session(session1)
            auditor.log_session_start(session1, {"model": "test_v1"})
            
            challenges1 = [f"challenge_{i}" for i in range(10)]
            for cid in challenges1:
                policy.record_use(cid, session1)
                auditor.log_challenge_use(session1, cid, False)
            
            stats1 = policy.end_session(session1)
            auditor.log_session_complete(session1, stats1)
            
            # Save state
            policy.save_state()
            
            # Session 2: Reuse some challenges
            session2 = "session_002"
            policy.start_session(session2)
            auditor.log_session_start(session2, {"model": "test_v2"})
            
            # Mix of new and reused challenges
            reused = challenges1[:3]  # Reuse 3
            fresh = [f"new_challenge_{i}" for i in range(7)]
            challenges2 = reused + fresh
            
            for cid in challenges2:
                is_reused = cid in reused
                policy.record_use(cid, session2)
                auditor.log_challenge_use(session2, cid, is_reused)
            
            # Check leakage
            is_safe, observed_rho = policy.check_leakage_threshold(challenges2)
            assert abs(observed_rho - 0.3) < 0.01  # 3/10 = 0.3
            assert not is_safe  # Over 0.25 threshold
            
            # Log violation
            auditor.log_policy_violation(
                session2,
                "leakage_threshold_exceeded",
                {"observed_rho": observed_rho, "threshold": 0.25}
            )
            
            stats2 = policy.end_session(session2)
            auditor.log_session_complete(session2, stats2)
            
            # Verify audit log
            with open(audit_path) as f:
                events = [json.loads(line) for line in f]
            
            # Should have events for both sessions
            session1_events = [e for e in events if e.get("session_id") == session1]
            session2_events = [e for e in events if e.get("session_id") == session2]
            
            assert len(session1_events) > 0
            assert len(session2_events) > 0
            
            # Should have policy violation for session2
            violations = [e for e in session2_events if e["event_type"] == "policy_violation"]
            assert len(violations) == 1
            assert violations[0]["violation_type"] == "leakage_threshold_exceeded"
    
    def test_recovery_after_crash(self):
        """Test recovery after simulated crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_path = str(Path(tmpdir) / "reuse_state.json")
            
            # Initial policy with some usage
            policy1 = ReusePolicy(u=3, rho_max=0.3, persistence_path=persistence_path)
            
            # Use challenges
            for i in range(20):
                cid = f"challenge_{i}"
                sid = f"session_{i % 4}"
                policy1.record_use(cid, sid)
            
            # Save state
            policy1.save_state()
            usage_before = dict(policy1.usage)
            
            # "Crash" - delete policy object
            del policy1
            
            # Recovery - create new policy and load
            policy2 = ReusePolicy(u=3, rho_max=0.3, persistence_path=persistence_path)
            policy2.load_state()
            
            # Should have recovered all usage
            assert len(policy2.usage) == len(usage_before)
            
            for cid, usage in usage_before.items():
                assert cid in policy2.usage
                assert policy2.usage[cid].use_count == usage.use_count
                
                # Continue using after recovery
                policy2.record_use(cid, "new_session")
                assert policy2.usage[cid].use_count == usage.use_count + 1


def test_challenge_usage_class():
    """Test ChallengeUsage dataclass."""
    usage = ChallengeUsage()
    
    # Initial state
    assert usage.use_count == 0
    assert usage.first_used is None
    assert usage.last_used is None
    assert len(usage.sessions) == 0
    
    # Record use
    usage.use_count = 1
    usage.first_used = time.time()
    usage.last_used = usage.first_used
    usage.sessions.add("session1")
    
    assert usage.use_count == 1
    assert usage.first_used is not None
    assert len(usage.sessions) == 1


def test_session_stats_class():
    """Test SessionStats dataclass."""
    stats = SessionStats()
    
    # Initial state
    assert stats.total_challenges == 0
    assert stats.leaked_challenges == 0
    assert stats.observed_rho == 0.0
    
    # Set values
    stats.total_challenges = 100
    stats.leaked_challenges = 25
    stats.observed_rho = 0.25
    
    assert stats.total_challenges == 100
    assert stats.leaked_challenges == 25
    assert abs(stats.observed_rho - 0.25) < 0.001


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])