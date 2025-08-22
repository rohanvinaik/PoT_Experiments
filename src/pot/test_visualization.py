#!/usr/bin/env python3
"""
Test script for visualization tools.
Verifies all visualization functions work correctly with mock data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import List

# Ensure we can import the modules
sys.path.insert(0, os.path.dirname(__file__))

from pot.core.visualize_sequential import (
    plot_verification_trajectory,
    plot_operating_characteristics,
    plot_anytime_validity, 
    VisualizationConfig
)


def create_mock_sprt_result(decision: str = 'H0', stopped_at: int = 50, 
                           true_mean: float = 0.03, tau: float = 0.05,
                           seed: int = 42):
    """Create a mock SPRTResult for testing"""
    np.random.seed(seed)
    
    trajectory = []
    running_sum = 0
    
    for i in range(1, stopped_at + 1):
        sample = np.random.normal(true_mean, 0.02)
        running_sum += sample
        mean = running_sum / i
        radius = 0.02 / np.sqrt(i) * 2.0  # Approximate 95% CI
        
        trajectory.append({
            'n': i,
            'mean': mean,
            'radius': radius,
            'tau': tau
        })
    
    class MockSPRTResult:
        def __init__(self):
            self.decision = decision
            self.stopped_at = stopped_at
            self.final_mean = trajectory[-1]['mean']
            self.confidence_radius = trajectory[-1]['radius']
            self.tau = tau
            self.trajectory = trajectory
            self.p_value = 0.023 if decision == 'H1' else 0.89
            self.forced_stop = False
    
    return MockSPRTResult()


def test_trajectory_plot():
    """Test single trajectory plotting"""
    print("Testing trajectory plotting...")
    
    try:
        # Create test result
        result = create_mock_sprt_result(decision='H0', stopped_at=45, true_mean=0.02)
        
        # Test basic plotting
        fig = plot_verification_trajectory(result, show_details=True)
        plt.close(fig)
        
        # Test with custom config
        config = VisualizationConfig(figsize=(10, 6), dpi=100)
        fig = plot_verification_trajectory(result, config=config, show_details=False)
        plt.close(fig)
        
        print("✓ Trajectory plotting works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Trajectory plotting failed: {e}")
        return False


def test_operating_characteristics():
    """Test operating characteristics plotting"""
    print("Testing operating characteristics plotting...")
    
    try:
        # Test with different parameters
        fig = plot_operating_characteristics(
            tau=0.05,
            alpha=0.05,
            beta=0.05,
            effect_sizes=[0.0, 0.02, 0.05, 0.08, 0.1],
            max_samples_fixed=500
        )
        plt.close(fig)
        
        print("✓ Operating characteristics plotting works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Operating characteristics plotting failed: {e}")
        return False


def test_anytime_validity():
    """Test anytime validity plotting"""
    print("Testing anytime validity plotting...")
    
    try:
        # Create multiple mock trajectories
        trajectories = []
        
        # H0 trajectories
        for i in range(10):
            result = create_mock_sprt_result(
                decision='H0', 
                stopped_at=30 + i * 5,
                true_mean=0.02,
                seed=100 + i
            )
            trajectories.append(result)
        
        # H1 trajectories
        for i in range(10):
            result = create_mock_sprt_result(
                decision='H1',
                stopped_at=20 + i * 3, 
                true_mean=0.08,
                seed=200 + i
            )
            trajectories.append(result)
        
        # Test plotting
        fig = plot_anytime_validity(trajectories, alpha=0.05)
        plt.close(fig)
        
        print("✓ Anytime validity plotting works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Anytime validity plotting failed: {e}")
        return False


def test_visualization_config():
    """Test VisualizationConfig functionality"""
    print("Testing VisualizationConfig...")
    
    try:
        # Test default config
        config1 = VisualizationConfig()
        assert config1.figsize == (12, 8)
        assert config1.dpi == 100
        
        # Test custom config
        config2 = VisualizationConfig(
            figsize=(8, 6),
            dpi=150,
            style='seaborn',
            show_grid=False
        )
        assert config2.figsize == (8, 6)
        assert config2.dpi == 150
        assert config2.show_grid == False
        
        print("✓ VisualizationConfig works correctly")
        return True
        
    except Exception as e:
        print(f"✗ VisualizationConfig failed: {e}")
        return False


def test_error_handling():
    """Test error handling in visualization functions"""
    print("Testing error handling...")
    
    try:
        # Test with empty trajectory
        class EmptyResult:
            def __init__(self):
                self.trajectory = []
                self.decision = 'H0'
                self.stopped_at = 0
        
        try:
            plot_verification_trajectory(EmptyResult())
            print("✗ Should have raised ValueError for empty trajectory")
            return False
        except ValueError:
            pass  # Expected
        
        # Test with empty trajectory list
        try:
            plot_anytime_validity([])
            print("✗ Should have raised ValueError for empty trajectory list")
            return False
        except ValueError:
            pass  # Expected
        
        print("✓ Error handling works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_matplotlib_compatibility():
    """Test matplotlib backend compatibility"""
    print("Testing matplotlib compatibility...")
    
    try:
        # Check matplotlib backend
        backend = plt.get_backend()
        print(f"  Matplotlib backend: {backend}")
        
        # Test basic plotting works
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        
        print("✓ Matplotlib compatibility verified")
        return True
        
    except Exception as e:
        print(f"✗ Matplotlib compatibility failed: {e}")
        return False


def run_all_tests():
    """Run all visualization tests"""
    print("🧪 Testing Sequential Verification Visualization Tools")
    print("=" * 60)
    
    tests = [
        test_matplotlib_compatibility,
        test_visualization_config,
        test_trajectory_plot,
        test_operating_characteristics,
        test_anytime_validity,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Visualization tools are working correctly.")
        print("\nNext steps:")
        print("1. Run: python pot/examples/visualization_examples.py")
        print("2. Install streamlit: pip install streamlit")
        print("3. Run interactive demo: streamlit run pot/core/visualize_sequential.py")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)