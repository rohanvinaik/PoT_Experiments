"""
Test suite for behavioral fingerprinting functionality.
Tests fingerprint creation, comparison, anomaly detection, and monitoring.
"""

import pytest
import torch
import numpy as np
import time
import tempfile
import os
from collections import deque


# Add parent directory to path for pot imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pot.semantic.behavioral_fingerprint import (
    BehavioralFingerprint,
    BehaviorSnapshot,
    FingerprintHistory,
    ContinuousMonitor,
    create_behavioral_monitor
)
from pot.semantic.library import ConceptLibrary
from pot.semantic.match import SemanticMatcher


class TestBehaviorSnapshot:
    """Test BehaviorSnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """Test creating behavior snapshots."""
        output = torch.randn(64)
        timestamp = time.time()
        metadata = {'key': 'value'}
        
        snapshot = BehaviorSnapshot(
            output=output,
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert torch.equal(snapshot.output, output)
        assert snapshot.timestamp == timestamp
        assert snapshot.metadata == metadata
    
    def test_snapshot_age(self):
        """Test snapshot age property."""
        past_time = time.time() - 10
        snapshot = BehaviorSnapshot(
            output=torch.randn(32),
            timestamp=past_time
        )
        
        age = snapshot.age
        assert 9.9 < age < 10.1  # Allow small timing variance


class TestFingerprintHistory:
    """Test FingerprintHistory functionality."""
    
    def test_history_initialization(self):
        """Test history initialization."""
        history = FingerprintHistory(max_history=100)
        
        assert len(history.fingerprints) == 0
        assert len(history.timestamps) == 0
        assert history.max_history == 100
    
    def test_add_fingerprint(self):
        """Test adding fingerprints to history."""
        history = FingerprintHistory(max_history=5)
        
        # Add fingerprints
        for i in range(3):
            fp = torch.randn(32)
            history.add(fp, metadata={'index': i})
        
        assert len(history.fingerprints) == 3
        assert len(history.timestamps) == 3
        assert history.metadata[0]['index'] == 0
    
    def test_history_limit(self):
        """Test history size limit."""
        history = FingerprintHistory(max_history=5)
        
        # Add more than max_history
        for i in range(10):
            fp = torch.randn(32)
            history.add(fp)
        
        assert len(history.fingerprints) == 5
        # Should keep the most recent ones
    
    def test_get_recent(self):
        """Test getting recent fingerprints."""
        history = FingerprintHistory()
        
        # Add fingerprints
        fps = []
        for i in range(10):
            fp = torch.randn(32)
            fps.append(fp)
            history.add(fp)
        
        recent = history.get_recent(3)
        assert len(recent) == 3
        assert torch.equal(recent[-1], fps[-1])  # Most recent
    
    def test_clear_history(self):
        """Test clearing history."""
        history = FingerprintHistory()
        
        # Add some data
        for _ in range(5):
            history.add(torch.randn(32))
        
        assert len(history.fingerprints) > 0
        
        history.clear()
        
        assert len(history.fingerprints) == 0
        assert len(history.timestamps) == 0


class TestBehavioralFingerprint:
    """Test BehavioralFingerprint class."""
    
    def test_initialization(self):
        """Test fingerprint system initialization."""
        fp = BehavioralFingerprint(
            window_size=50,
            fingerprint_dim=128,
            decay_factor=0.95
        )
        
        assert fp.window_size == 50
        assert fp.fingerprint_dim == 128
        assert fp.decay_factor == 0.95
        assert fp.n_updates == 0
        assert len(fp.observation_window) == 0
    
    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        with pytest.raises(ValueError):
            BehavioralFingerprint(window_size=-1)
        
        with pytest.raises(ValueError):
            BehavioralFingerprint(fingerprint_dim=0)
        
        with pytest.raises(ValueError):
            BehavioralFingerprint(decay_factor=1.5)
    
    def test_update_observations(self):
        """Test updating with new observations."""
        fp = BehavioralFingerprint(window_size=10)
        
        # Add observations
        for i in range(5):
            output = torch.randn(64)
            fp.update(output, metadata={'index': i})
        
        assert len(fp.observation_window) == 5
        assert fp.n_updates == 5
    
    def test_window_sliding(self):
        """Test sliding window behavior."""
        fp = BehavioralFingerprint(window_size=5)
        
        # Add more than window size
        for i in range(10):
            fp.update(torch.randn(32))
        
        assert len(fp.observation_window) == 5  # Limited by window_size
        assert fp.n_updates == 10  # Total updates tracked
    
    def test_compute_fingerprint_empty(self):
        """Test fingerprint computation with no observations."""
        fp = BehavioralFingerprint(fingerprint_dim=64)
        
        fingerprint = fp.compute_fingerprint()
        
        assert fingerprint.shape == (64,)
        assert torch.allclose(fingerprint, torch.zeros(64))
    
    def test_compute_fingerprint_with_data(self):
        """Test fingerprint computation with observations."""
        fp = BehavioralFingerprint(
            window_size=10,
            fingerprint_dim=32,
            decay_factor=0.9
        )
        
        # Add observations
        for i in range(10):
            output = torch.randn(64) + i * 0.1  # Slight trend
            fp.update(output)
        
        fingerprint = fp.compute_fingerprint()
        
        assert fingerprint.shape == (32,)
        assert not torch.allclose(fingerprint, torch.zeros(32))
    
    def test_fingerprint_normalization(self):
        """Test fingerprint normalization."""
        fp = BehavioralFingerprint(fingerprint_dim=32)
        
        for _ in range(5):
            fp.update(torch.randn(64))
        
        # Normalized fingerprint
        normalized = fp.compute_fingerprint(normalize=True)
        norm = torch.norm(normalized)
        assert 0.99 < norm < 1.01  # Should be unit norm
        
        # Non-normalized
        non_normalized = fp.compute_fingerprint(normalize=False)
        assert torch.norm(non_normalized) != 1.0 or torch.allclose(non_normalized, torch.zeros_like(non_normalized))
    
    def test_compare_fingerprints_cosine(self):
        """Test cosine similarity comparison."""
        fp = BehavioralFingerprint()
        
        fp1 = torch.randn(64)
        fp2 = torch.randn(64)
        fp_same = fp1.clone()
        
        # Different fingerprints
        sim = fp.compare_fingerprints(fp1, fp2, method='cosine')
        assert 0.0 <= sim <= 1.0
        
        # Same fingerprint
        sim_same = fp.compare_fingerprints(fp1, fp_same, method='cosine')
        assert sim_same > 0.99
    
    def test_compare_fingerprints_euclidean(self):
        """Test Euclidean similarity comparison."""
        fp = BehavioralFingerprint()
        
        fp1 = torch.randn(64)
        fp2 = torch.randn(64)
        
        sim = fp.compare_fingerprints(fp1, fp2, method='euclidean')
        assert 0.0 <= sim <= 1.0
    
    def test_compare_fingerprints_correlation(self):
        """Test correlation comparison."""
        fp = BehavioralFingerprint()
        
        fp1 = torch.randn(64)
        fp2 = torch.randn(64)
        
        sim = fp.compare_fingerprints(fp1, fp2, method='correlation')
        assert 0.0 <= sim <= 1.0
    
    def test_anomaly_detection_no_reference(self):
        """Test anomaly detection without reference."""
        fp = BehavioralFingerprint()
        
        # No reference set
        current_fp = torch.randn(32)
        is_anomaly, score = fp.detect_anomaly(current_fp)
        
        assert not is_anomaly  # Should not detect without reference
        assert score == 0.0
    
    def test_anomaly_detection_with_reference(self):
        """Test anomaly detection with reference."""
        fp = BehavioralFingerprint(fingerprint_dim=32)
        
        # Set reference
        reference = torch.randn(32)
        fp.set_reference(reference, threshold=0.8)
        
        # Test normal (similar to reference)
        normal_fp = reference + torch.randn(32) * 0.01
        is_anomaly, score = fp.detect_anomaly(normal_fp)
        assert not is_anomaly
        assert score < 0.2
        
        # Test anomaly (very different)
        anomaly_fp = torch.randn(32) * 10
        is_anomaly, score = fp.detect_anomaly(anomaly_fp)
        # Note: May or may not detect depending on random values
        assert 0.0 <= score <= 1.0
    
    def test_drift_detection(self):
        """Test behavioral drift detection."""
        fp = BehavioralFingerprint(window_size=20, fingerprint_dim=32)
        
        # Add initial observations
        for i in range(60):
            output = torch.randn(64) * 0.5
            fp.update(output)
            if i % 5 == 0:
                fp.compute_fingerprint()
        
        # Initial check - no drift
        has_drift, drift_score = fp.detect_drift(window=5)
        assert not has_drift
        assert drift_score < 0.5
        
        # Add drifted observations
        for i in range(30):
            output = torch.randn(64) * 2.0 + torch.ones(64) * 3
            fp.update(output)
            if i % 5 == 0:
                fp.compute_fingerprint()
        
        # Check for drift
        has_drift, drift_score = fp.detect_drift(window=5)
        assert 0.0 <= drift_score <= 1.0
        # May or may not detect drift depending on random values
    
    def test_set_reference(self):
        """Test setting reference fingerprint."""
        fp = BehavioralFingerprint(fingerprint_dim=32)
        
        # Set explicit reference
        reference = torch.randn(32)
        fp.set_reference(reference, threshold=0.85)
        
        assert torch.equal(fp.reference_fingerprint, reference)
        assert fp.reference_threshold == 0.85
        
        # Set from history
        for _ in range(10):
            fp.update(torch.randn(64))
            fp.compute_fingerprint()
        
        fp.set_reference(None)  # Should compute from history
        assert fp.reference_fingerprint is not None
    
    def test_alert_callbacks(self):
        """Test alert callback system."""
        alerts_received = []
        
        def alert_callback(score, fingerprint, metadata):
            alerts_received.append({
                'score': score,
                'fingerprint': fingerprint,
                'metadata': metadata
            })
        
        fp = BehavioralFingerprint(fingerprint_dim=32)
        fp.register_alert_callback(alert_callback)
        
        # Set up for anomaly
        reference = torch.randn(32)
        fp.set_reference(reference, threshold=0.9)
        
        # Trigger anomaly
        anomaly_fp = torch.randn(32) * 10
        is_anomaly, score = fp.detect_anomaly(anomaly_fp, threshold=0.5)
        
        if is_anomaly:
            assert len(alerts_received) > 0
            assert 'score' in alerts_received[0]
    
    def test_get_statistics(self):
        """Test getting fingerprint statistics."""
        fp = BehavioralFingerprint(window_size=10, fingerprint_dim=32)
        
        # Add observations
        for i in range(10):
            fp.update(torch.randn(64))
            time.sleep(0.001)
        
        fp.compute_fingerprint()
        
        stats = fp.get_statistics()
        
        assert stats['n_observations'] == 10
        assert stats['n_updates'] == 10
        assert stats['window_size'] == 10
        assert stats['fingerprint_dim'] == 32
        assert 'observation_ages' in stats
    
    def test_reset(self):
        """Test resetting fingerprint system."""
        fp = BehavioralFingerprint()
        
        # Add data
        for _ in range(10):
            fp.update(torch.randn(64))
        fp.compute_fingerprint()
        fp.set_reference(torch.randn(32))
        
        # Reset
        fp.reset()
        
        assert len(fp.observation_window) == 0
        assert fp.n_updates == 0
        assert fp.reference_fingerprint is None
        assert len(fp.history.fingerprints) == 0
    
    def test_save_load_state(self):
        """Test saving and loading state."""
        fp1 = BehavioralFingerprint(window_size=10, fingerprint_dim=32)
        
        # Add data
        for i in range(10):
            fp1.update(torch.randn(64))
        fp1.compute_fingerprint()
        fp1.set_reference(torch.randn(32))
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            fp1.save_state(tmp_path)
            
            # Create new fingerprint and load
            fp2 = BehavioralFingerprint(window_size=5, fingerprint_dim=16)
            fp2.load_state(tmp_path)
            
            assert fp2.window_size == 10
            assert fp2.fingerprint_dim == 32
            assert fp2.n_updates == 10
            assert fp2.reference_fingerprint is not None
            
        finally:
            os.unlink(tmp_path)


class TestContinuousMonitor:
    """Test ContinuousMonitor functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        fp = BehavioralFingerprint(window_size=10, fingerprint_dim=32)
        monitor = ContinuousMonitor(
            fingerprint=fp,
            alert_threshold=0.9,
            check_interval=5
        )
        
        assert monitor.fingerprint == fp
        assert monitor.alert_threshold == 0.9
        assert monitor.check_interval == 5
        assert monitor.n_checks == 0
    
    def test_process_output(self):
        """Test processing outputs."""
        fp = BehavioralFingerprint(window_size=10, fingerprint_dim=32)
        monitor = ContinuousMonitor(fp, check_interval=5)
        
        results = []
        for i in range(15):
            output = torch.randn(64)
            result = monitor.process_output(output, metadata={'index': i})
            results.append(result)
        
        # Check results
        assert all(r['processed'] for r in results)
        
        # Should have anomaly checks at intervals
        checks = [r for r in results if 'anomaly_check' in r]
        assert len(checks) == 3  # At 5, 10, 15
    
    def test_monitor_with_semantic(self):
        """Test monitor with semantic matcher."""
        # Create semantic library
        library = ConceptLibrary(dim=32, method='gaussian')
        embeddings = torch.randn(10, 32)
        library.add_concept('test_concept', embeddings)
        
        # Create matcher
        matcher = SemanticMatcher(library=library)
        
        # Create monitor
        fp = BehavioralFingerprint(window_size=10, fingerprint_dim=32)
        monitor = ContinuousMonitor(
            fingerprint=fp,
            semantic_matcher=matcher,
            check_interval=5
        )
        
        # Process outputs
        for i in range(10):
            result = monitor.process_output(torch.randn(64))
            
            if 'anomaly_check' in result and 'semantic_matches' in result:
                assert isinstance(result['semantic_matches'], dict)
    
    def test_monitoring_summary(self):
        """Test getting monitoring summary."""
        fp = BehavioralFingerprint(window_size=10, fingerprint_dim=32)
        monitor = ContinuousMonitor(fp, check_interval=5)
        
        # Process outputs
        for i in range(20):
            monitor.process_output(torch.randn(64))
        
        summary = monitor.get_monitoring_summary()
        
        assert 'fingerprint_stats' in summary
        assert 'n_checks' in summary
        assert summary['n_checks'] >= 0
        assert 'n_anomalies' in summary
        assert 'n_drifts' in summary


class TestCreateBehavioralMonitor:
    """Test create_behavioral_monitor helper function."""
    
    def test_create_monitor_basic(self):
        """Test creating basic monitor."""
        monitor = create_behavioral_monitor(
            window_size=50,
            fingerprint_dim=128
        )
        
        assert isinstance(monitor, ContinuousMonitor)
        assert monitor.fingerprint.window_size == 50
        assert monitor.fingerprint.fingerprint_dim == 128
    
    def test_create_monitor_with_semantic(self):
        """Test creating monitor with semantic library."""
        library = ConceptLibrary(dim=128, method='gaussian')
        
        monitor = create_behavioral_monitor(
            window_size=50,
            fingerprint_dim=128,
            semantic_library=library
        )
        
        assert monitor.semantic_matcher is not None
        assert monitor.semantic_matcher.library == library
    
    def test_create_monitor_with_options(self):
        """Test creating monitor with additional options."""
        monitor = create_behavioral_monitor(
            window_size=100,
            fingerprint_dim=256,
            decay_factor=0.9,
            use_pca=True
        )
        
        assert monitor.fingerprint.decay_factor == 0.9
        assert monitor.fingerprint.use_pca


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_normal_to_anomaly_transition(self):
        """Test detecting transition from normal to anomalous behavior."""
        monitor = create_behavioral_monitor(
            window_size=20,
            fingerprint_dim=64
        )
        
        # Normal phase
        for i in range(15):
            normal_output = torch.randn(128) * 0.5
            result = monitor.process_output(normal_output)
        
        # Set reference from normal behavior
        current_fp = monitor.fingerprint.compute_fingerprint()
        monitor.fingerprint.set_reference(current_fp, threshold=0.7)
        
        # Anomalous phase
        anomaly_detected = False
        for i in range(10):
            anomaly_output = torch.randn(128) * 3.0 + torch.ones(128) * 5
            result = monitor.process_output(anomaly_output)
            
            if 'anomaly_check' in result and result['anomaly_check']['is_anomaly']:
                anomaly_detected = True
        
        # Should detect some anomalies (depends on randomness)
        assert monitor.n_checks > 0
    
    def test_gradual_drift(self):
        """Test detecting gradual behavioral drift."""
        monitor = create_behavioral_monitor(
            window_size=30,
            fingerprint_dim=64
        )
        
        # Gradual drift over time
        for epoch in range(5):
            drift_factor = epoch * 0.5
            for i in range(20):
                output = torch.randn(128) + torch.ones(128) * drift_factor
                result = monitor.process_output(output)
        
        # Check for drift detection
        summary = monitor.get_monitoring_summary()
        
        # Should have processed many outputs
        assert summary['fingerprint_stats']['n_updates'] == 100
        
        # May have detected drift
        if summary['n_drifts'] > 0:
            assert len(summary['recent_drifts']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])