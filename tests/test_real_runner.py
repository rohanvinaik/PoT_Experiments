import pytest
import numpy as np
import json
from pathlib import Path
import tempfile
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.core.sequential_runner import TestMetrics, SequentialTestRunner
from pot.utils.json_utils import NumpyJSONEncoder, safe_json_dump, validate_no_mocks, separate_mock_files

def test_numpy_json_encoder():
    """Test that numpy types are properly converted"""
    
    data = {
        "bool_np": np.bool_(True),
        "int_np": np.int32(42),
        "float_np": np.float64(3.14),
        "array_np": np.array([1, 2, 3]),
        "nested": {
            "bool_np": np.bool_(False),
            "float_np": np.float32(2.71)
        }
    }
    
    # Should not raise
    json_str = json.dumps(data, cls=NumpyJSONEncoder)
    parsed = json.loads(json_str)
    
    # Check types are correct
    assert isinstance(parsed["bool_np"], bool)
    assert isinstance(parsed["int_np"], int)
    assert isinstance(parsed["float_np"], float)
    assert isinstance(parsed["array_np"], list)
    assert isinstance(parsed["nested"]["bool_np"], bool)
    print("✅ Numpy JSON encoder test passed")

def test_metrics_json_safe():
    """Test that TestMetrics converts to JSON-safe dict"""
    
    metrics = TestMetrics(
        alpha=0.01,
        beta=0.01,
        n_used=100,
        n_max=500,
        statistic_mean=np.float32(0.03),  # numpy type
        statistic_var=np.float64(0.001),  # numpy type
        boundary_type="EB",
        decision="accept_id",
        hypothesis="H1",
        stopping_time=100,
        t_load=1.5,
        t_infer_total=10.2,
        t_per_query=0.102,
        t_setup=0.5,
        t_scoring=2.1,
        t_total=14.3,
        per_query_times=[np.float32(0.1)] * 10,  # numpy types
        per_query_scores=[np.float64(0.02)] * 10,  # numpy types
        confidence_intervals=[(0.01, 0.05)] * 10
    )
    
    # Convert to JSON-safe dict
    safe_dict = metrics.to_json_safe_dict()
    
    # Should be JSON serializable
    json_str = json.dumps(safe_dict)
    parsed = json.loads(json_str)
    
    # Check conversions
    assert isinstance(parsed["statistic_mean"], float)
    assert isinstance(parsed["statistic_var"], float)
    assert all(isinstance(t, float) for t in parsed["per_query_times"])
    print("✅ TestMetrics JSON-safe conversion test passed")

def test_validate_no_mocks():
    """Test mock detection in data"""
    
    # Real data - should pass
    real_data = {
        "model": "gpt2",
        "score": 0.95,
        "results": [0.1, 0.2, 0.3]
    }
    assert validate_no_mocks(real_data) == True
    
    # Mock data - should fail
    mock_data = {
        "model": "mock_model",
        "score": 0.5,
        "results": ["placeholder", "dummy_value"]
    }
    assert validate_no_mocks(mock_data) == False
    
    # Nested mock - should fail
    nested_mock = {
        "model": "real_model",
        "config": {
            "path": "/path/to/fake_model"
        }
    }
    assert validate_no_mocks(nested_mock) == False
    
    print("✅ Mock validation test passed")

def test_separate_mock_files():
    """Test that mock data goes to separate files"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Real data should go to normal file
        real_data = {"type": "real", "value": 42}
        real_file = tmppath / "results.json"
        safe_json_dump(real_data, real_file)
        assert real_file.exists()
        
        # Mock data should get _mock suffix if detected
        mock_data = {"type": "mock_test", "value": 0}
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_path = separate_mock_files(
                mock_data, 
                tmppath / "results2.json",
                force_real=False
            )
        
        # Should have created mock file
        assert "_mock" in str(result_path)
        print("✅ Mock file separation test passed")

def test_no_mock_in_real_test():
    """Ensure real test runner never produces mock data"""
    
    # This would be an integration test with actual models
    # For unit test, just verify the runner doesn't have mock generation
    
    import inspect
    from pot.core import sequential_runner
    
    source = inspect.getsource(sequential_runner)
    
    # Check that mock-related terms don't appear in production code
    mock_terms = ["mock", "fake", "dummy", "placeholder"]
    found_issues = []
    
    lines = source.split('\n')
    for i, line in enumerate(lines):
        # Skip comments and docstrings
        if line.strip().startswith('#') or '"""' in line or "'''" in line:
            continue
            
        for term in mock_terms:
            if term in line.lower():
                # Check it's not in a string for error messages or logging
                if f'"{term}' not in line and f"'{term}" not in line:
                    # Allow "mock" in validate_no_mocks function name
                    if "validate_no_mocks" not in line and "separate_mock_files" not in line:
                        found_issues.append(f"Line {i+1}: {line.strip()}")
    
    if found_issues:
        print(f"⚠️ Found potential mock code in production:")
        for issue in found_issues[:3]:  # Show first 3
            print(f"   {issue}")
    
    # We expect no mock-related code in production
    assert len(found_issues) == 0, f"Found {len(found_issues)} lines with mock-related terms"
    print("✅ No mock code in production test passed")

def test_json_edge_cases():
    """Test JSON serialization edge cases"""
    
    # Test with various numpy dtypes
    edge_cases = {
        "np_bool": np.bool_(True),
        "np_int8": np.int8(127),
        "np_int16": np.int16(32767),
        "np_int32": np.int32(2147483647),
        "np_int64": np.int64(9223372036854775807),
        "np_uint8": np.uint8(255),
        "np_float16": np.float16(3.14),
        "np_float32": np.float32(3.14159),
        "np_float64": np.float64(3.141592653589793),
        "np_array_1d": np.array([1, 2, 3]),
        "np_array_2d": np.array([[1, 2], [3, 4]]),
        "np_array_bool": np.array([True, False, True]),
        "path": Path("/tmp/test"),
        "bytes": b"test_bytes"
    }
    
    # Should serialize without error
    json_str = json.dumps(edge_cases, cls=NumpyJSONEncoder)
    parsed = json.loads(json_str)
    
    # Check conversions
    assert isinstance(parsed["np_bool"], bool)
    assert isinstance(parsed["np_int32"], int)
    assert isinstance(parsed["np_float64"], float)
    assert isinstance(parsed["np_array_1d"], list)
    assert isinstance(parsed["np_array_2d"], list)
    assert parsed["path"] == "/tmp/test"
    assert parsed["bytes"] == "746573745f6279746573"  # hex representation
    
    print("✅ JSON edge cases test passed")

def test_metrics_with_empty_data():
    """Test metrics with empty or minimal data"""
    
    metrics = TestMetrics(
        alpha=0.01,
        beta=0.01,
        n_used=0,
        n_max=100,
        statistic_mean=0.0,
        statistic_var=0.0,
        boundary_type="EB",
        decision="undecided",
        hypothesis="inconclusive",
        stopping_time=0,
        t_load=0.0,
        t_infer_total=0.0,
        t_per_query=0.0,
        t_setup=0.0,
        t_scoring=0.0,
        t_total=0.0,
        per_query_times=[],
        per_query_scores=[],
        confidence_intervals=[]
    )
    
    # Should handle empty lists
    safe_dict = metrics.to_json_safe_dict()
    json_str = json.dumps(safe_dict)
    parsed = json.loads(json_str)
    
    assert parsed["per_query_times"] == []
    assert parsed["per_query_scores"] == []
    assert parsed["confidence_intervals"] == []
    
    print("✅ Empty metrics test passed")

def test_safe_json_dump_creates_dirs():
    """Test that safe_json_dump creates directories if needed"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Nested path that doesn't exist
        nested_file = tmppath / "subdir1" / "subdir2" / "results.json"
        
        data = {"test": "value"}
        safe_json_dump(data, nested_file)
        
        assert nested_file.exists()
        assert nested_file.parent.exists()
        
        # Read back and verify
        with open(nested_file) as f:
            loaded = json.load(f)
        assert loaded == data
        
    print("✅ Directory creation test passed")

def run_all_tests():
    """Run all real runner tests"""
    print("\n🧪 Running Real Sequential Runner Tests\n")
    
    test_functions = [
        test_numpy_json_encoder,
        test_metrics_json_safe,
        test_validate_no_mocks,
        test_separate_mock_files,
        test_no_mock_in_real_test,
        test_json_edge_cases,
        test_metrics_with_empty_data,
        test_safe_json_dump_creates_dirs
    ]
    
    for test_func in test_functions:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✅ All real runner tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)