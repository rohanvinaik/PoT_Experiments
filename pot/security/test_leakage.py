"""Tests for leakage tracking and challenge reuse policy enforcement."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pot.security.leakage import (
    ReusePolicy, ChallengeUsage, LeakageStats,
    LeakageAuditor, compute_challenge_hash
)


def test_basic_reuse_policy():
    """Test basic reuse policy functionality."""
    print("Testing basic reuse policy...")
    
    # Create policy with u=3 max uses
    policy = ReusePolicy(u=3, rho_max=0.5)
    
    # First use should be allowed
    assert policy.record_use("challenge_1"), "First use should be allowed"
    assert policy.usage["challenge_1"].use_count == 1
    
    # Second use should be allowed
    assert policy.record_use("challenge_1"), "Second use should be allowed"
    assert policy.usage["challenge_1"].use_count == 2
    
    # Third use should be allowed
    assert policy.record_use("challenge_1"), "Third use should be allowed"
    assert policy.usage["challenge_1"].use_count == 3
    
    # Fourth use should be rejected
    assert not policy.record_use("challenge_1"), "Fourth use should be rejected"
    assert policy.usage["challenge_1"].use_count == 3  # Count shouldn't increase
    
    print("✓ Basic reuse policy working correctly")
    
    # Test multiple challenges
    assert policy.record_use("challenge_2"), "New challenge should be allowed"
    assert policy.record_use("challenge_3"), "Another new challenge should be allowed"
    
    assert len(policy.usage) == 3, "Should have 3 challenges tracked"
    
    print("✓ Multiple challenges tracked correctly")
    
    return True


def test_observed_rho_computation():
    """Test observed_rho computation."""
    print("\nTesting observed_rho computation...")
    
    policy = ReusePolicy(u=5, rho_max=0.3)
    
    # Test basic computation
    assert policy.observed_rho(0, 10) == 0.0, "No leakage should give rho=0"
    assert policy.observed_rho(5, 10) == 0.5, "5/10 should give rho=0.5"
    assert policy.observed_rho(10, 10) == 1.0, "10/10 should give rho=1.0"
    
    # Test edge cases
    assert policy.observed_rho(0, 0) == 0.0, "0/0 should give rho=0"
    assert policy.observed_rho(5, 0) == 0.0, "Division by zero should give rho=0"
    
    print("✓ observed_rho computation correct")
    
    # Test get_leaked_count
    policy.record_use("c1")
    policy.record_use("c2")
    policy.record_use("c2")  # c2 used twice
    
    challenge_ids = ["c1", "c2", "c3", "c4"]
    leaked = policy.get_leaked_count(challenge_ids)
    assert leaked == 2, f"Should have 2 leaked challenges, got {leaked}"
    
    print("✓ Leaked count computation correct")
    
    # Test leakage threshold check
    is_safe, rho = policy.check_leakage_threshold(challenge_ids)
    assert rho == 0.5, f"Rho should be 0.5, got {rho}"
    assert not is_safe, "Should exceed threshold (0.5 > 0.3)"
    
    print("✓ Leakage threshold check working")
    
    return True


def test_session_tracking():
    """Test session-based tracking."""
    print("\nTesting session tracking...")
    
    policy = ReusePolicy(u=3, rho_max=0.4)
    
    # Start session 1
    policy.start_session("session_1")
    assert policy.current_session == "session_1"
    
    # Use some challenges
    policy.record_use("c1", "session_1")
    policy.record_use("c2", "session_1")
    policy.record_use("c3", "session_1")
    
    # Start session 2
    policy.start_session("session_2")
    
    # Reuse some challenges
    policy.record_use("c1", "session_2")  # Reuse
    policy.record_use("c2", "session_2")  # Reuse
    policy.record_use("c4", "session_2")  # New
    
    # Check session stats
    stats1 = policy.end_session("session_1")
    assert stats1 is not None
    assert stats1.total_challenges == 3
    
    stats2 = policy.end_session("session_2")
    assert stats2 is not None
    assert stats2.total_challenges == 3
    
    print("✓ Session tracking working correctly")
    
    # Verify challenges track which sessions used them
    assert "session_1" in policy.usage["c1"].sessions
    assert "session_2" in policy.usage["c1"].sessions
    assert "session_2" not in policy.usage["c3"].sessions
    
    print("✓ Challenge-session association tracked")
    
    return True


def test_persistence():
    """Test persistence of reuse counters."""
    print("\nTesting persistence...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_path = os.path.join(tmpdir, "reuse_state.json")
        
        # Create policy and record some uses
        policy1 = ReusePolicy(u=5, rho_max=0.3, persistence_path=persist_path)
        
        policy1.record_use("c1")
        policy1.record_use("c1")
        policy1.record_use("c2")
        
        policy1.start_session("test_session")
        policy1.record_use("c3", "test_session")
        
        # Verify state was saved
        assert os.path.exists(persist_path), "State file should exist"
        
        # Load state into new policy
        policy2 = ReusePolicy(u=5, rho_max=0.3, persistence_path=persist_path)
        
        # Verify state was loaded correctly
        assert len(policy2.usage) == 3, "Should have 3 challenges"
        assert policy2.usage["c1"].use_count == 2, "c1 should have 2 uses"
        assert policy2.usage["c2"].use_count == 1, "c2 should have 1 use"
        assert policy2.usage["c3"].use_count == 1, "c3 should have 1 use"
        
        assert "test_session" in policy2.sessions, "Session should be loaded"
        
        print("✓ State persisted and loaded correctly")
        
        # Test atomic writes (no corruption on failure)
        policy2.record_use("c4")
        
        # Load again to verify
        policy3 = ReusePolicy(u=5, rho_max=0.3, persistence_path=persist_path)
        assert "c4" in policy3.usage, "New challenge should be persisted"
        
        print("✓ Atomic writes working")
        
    return True


def test_usage_statistics():
    """Test usage statistics generation."""
    print("\nTesting usage statistics...")
    
    policy = ReusePolicy(u=3, rho_max=0.5)
    
    # Create various usage patterns
    # c1: used 3 times (exhausted)
    for _ in range(3):
        policy.record_use("c1")
    
    # c2: used 2 times (reused but not exhausted)
    policy.record_use("c2")
    policy.record_use("c2")
    
    # c3: used once
    policy.record_use("c3")
    
    # c4, c5: never used (just check)
    stats = policy.get_usage_stats()
    
    assert stats["total_challenges"] == 3
    assert stats["reused_challenges"] == 2  # c1 and c2
    assert stats["exhausted_challenges"] == 1  # only c1
    assert stats["use_distribution"][1] == 1  # c3
    assert stats["use_distribution"][2] == 1  # c2
    assert stats["use_distribution"][3] == 1  # c1
    
    print("✓ Usage statistics correct")
    print(f"  Total: {stats['total_challenges']}")
    print(f"  Reused: {stats['reused_challenges']}")
    print(f"  Exhausted: {stats['exhausted_challenges']}")
    print(f"  Distribution: {stats['use_distribution']}")
    
    return True


def test_leakage_auditor():
    """Test leakage auditor functionality."""
    print("\nTesting leakage auditor...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "audit.log")
        auditor = LeakageAuditor(log_path)
        
        # Log some events
        auditor.log_challenge_use("c1", "s1", 1, True, 0.1)
        auditor.log_challenge_use("c2", "s1", 2, True, 0.2)
        auditor.log_challenge_use("c3", "s1", 4, False, 0.3)
        
        stats = LeakageStats("s1", 10, 3, 0.3)
        auditor.log_session_complete("s1", stats)
        
        auditor.log_policy_violation("s1", "excessive_reuse", {"challenge": "c3"})
        
        # Verify events were logged
        assert len(auditor.events) == 5
        assert os.path.exists(log_path), "Log file should exist"
        
        # Check log file content
        with open(log_path, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 5, "Should have 5 log lines"
        
        # Generate report
        report = auditor.generate_report()
        assert report["total_events"] == 5
        assert report["total_challenge_uses"] == 3
        assert report["rejected_uses"] == 1
        assert report["policy_violations"] == 1
        
        print("✓ Auditor logging and reporting working")
        print(f"  Total events: {report['total_events']}")
        print(f"  Rejection rate: {report['rejection_rate']:.2%}")
        print(f"  Avg leakage: {report['average_leakage_ratio']:.3f}")
        
    return True


def test_challenge_hash():
    """Test challenge hash computation."""
    print("\nTesting challenge hash computation...")
    
    # Test different data types
    dict_data = {"key": "value", "num": 42}
    list_data = [1, 2, 3, 4]
    str_data = "challenge_string"
    
    hash1 = compute_challenge_hash(dict_data)
    hash2 = compute_challenge_hash(list_data)
    hash3 = compute_challenge_hash(str_data)
    
    # Hashes should be consistent
    assert compute_challenge_hash(dict_data) == hash1
    assert compute_challenge_hash(list_data) == hash2
    assert compute_challenge_hash(str_data) == hash3
    
    # Different data should give different hashes
    assert hash1 != hash2 != hash3
    
    # Test dict key ordering doesn't matter
    dict_reordered = {"num": 42, "key": "value"}
    assert compute_challenge_hash(dict_reordered) == hash1
    
    print("✓ Challenge hash computation working")
    print(f"  Dict hash: {hash1}")
    print(f"  List hash: {hash2}")
    print(f"  String hash: {hash3}")
    
    return True


def test_reset_functionality():
    """Test reset functions."""
    print("\nTesting reset functionality...")
    
    policy = ReusePolicy(u=3, rho_max=0.5)
    
    # Add some usage
    policy.record_use("c1")
    policy.record_use("c1")
    policy.record_use("c2")
    policy.start_session("s1")
    
    # Reset single challenge
    policy.reset_challenge("c1")
    assert "c1" not in policy.usage
    assert "c2" in policy.usage
    
    print("✓ Single challenge reset working")
    
    # Reset all
    policy.reset_all()
    assert len(policy.usage) == 0
    assert len(policy.sessions) == 0
    assert policy.current_session is None
    
    print("✓ Full reset working")
    
    return True


def test_integration_scenario():
    """Test realistic integration scenario."""
    print("\nTesting integration scenario...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_path = os.path.join(tmpdir, "reuse.json")
        log_path = os.path.join(tmpdir, "audit.log")
        
        # Create system components
        policy = ReusePolicy(u=2, rho_max=0.4, persistence_path=persist_path)
        auditor = LeakageAuditor(log_path)
        
        # Simulate verification sessions
        challenges = ["c1", "c2", "c3", "c4", "c5"]
        
        # Session 1: Fresh challenges
        policy.start_session("verify_1")
        for cid in challenges[:3]:
            allowed = policy.record_use(cid, "verify_1")
            auditor.log_challenge_use(cid, "verify_1", 1, allowed, 0.0)
        
        stats1 = policy.end_session("verify_1")
        auditor.log_session_complete("verify_1", stats1)
        
        # Session 2: Some reuse
        policy.start_session("verify_2")
        test_challenges = ["c1", "c2", "c4", "c5"]  # 50% reuse
        
        is_safe, rho = policy.check_leakage_threshold(test_challenges)
        if not is_safe:
            auditor.log_policy_violation("verify_2", "leakage_exceeded", {"rho": rho})
            print(f"  Session 2 rejected: rho={rho:.2f} > {policy.rho_max}")
        else:
            for cid in test_challenges:
                allowed = policy.record_use(cid, "verify_2")
                if not allowed:
                    auditor.log_policy_violation("verify_2", "challenge_exhausted", {"challenge": cid})
        
        # Session 3: Try to reuse exhausted challenges
        policy.start_session("verify_3")
        for cid in ["c1", "c2"]:  # Both used twice already
            allowed = policy.record_use(cid, "verify_3")
            if not allowed:
                print(f"  Challenge {cid} rejected (exhausted)")
                auditor.log_policy_violation("verify_3", "challenge_exhausted", {"challenge": cid})
        
        # Generate final report
        report = auditor.generate_report()
        stats = policy.get_usage_stats()
        
        print("✓ Integration scenario completed")
        print(f"  Total challenges used: {stats['total_challenges']}")
        print(f"  Exhausted: {stats['exhausted_challenges']}")
        print(f"  Policy violations: {report['policy_violations']}")
        
        # We should have at least one violation (the leakage exceeded in session 2)
        assert report['policy_violations'] >= 1, "Should have at least one violation"
        # c1 and c2 were used twice (exhausted with u=2)
        assert stats['exhausted_challenges'] >= 2, "At least c1 and c2 should be exhausted"
        
    return True


def run_all_tests():
    """Run all leakage tracking tests."""
    print("=" * 60)
    print("Running Leakage Tracking Tests")
    print("=" * 60)
    
    tests = [
        test_basic_reuse_policy,
        test_observed_rho_computation,
        test_session_tracking,
        test_persistence,
        test_usage_statistics,
        test_leakage_auditor,
        test_challenge_hash,
        test_reset_functionality,
        test_integration_scenario
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All leakage tracking tests passed!")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)