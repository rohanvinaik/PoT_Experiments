#!/usr/bin/env python3
"""Demo showing real sequential testing with JSON serialization"""

import sys
import os
import json
import numpy as np
from pathlib import Path
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.core.sequential_runner import TestMetrics
from pot.utils.json_utils import NumpyJSONEncoder, safe_json_dump, validate_no_mocks

def main():
    print("🎯 Real Sequential Testing & JSON Serialization Demo\n")
    print("=" * 60)
    
    # Demo 1: TestMetrics with numpy types
    print("\n1️⃣ TestMetrics JSON Serialization")
    print("-" * 40)
    
    metrics = TestMetrics(
        alpha=0.01,
        beta=0.01,
        n_used=50,
        n_max=100,
        statistic_mean=np.float32(0.045),  # numpy type
        statistic_var=np.float64(0.002),   # numpy type
        boundary_type="EB",
        decision="accept_id",
        hypothesis="H1",
        stopping_time=50,
        t_load=2.5,
        t_infer_total=15.3,
        t_per_query=0.306,
        t_setup=0.8,
        t_scoring=3.2,
        t_total=21.8,
        per_query_times=[np.float32(0.3 + i*0.01) for i in range(5)],
        per_query_scores=[np.float64(0.04 + i*0.002) for i in range(5)],
        confidence_intervals=[(0.03 + i*0.001, 0.06 - i*0.001) for i in range(5)]
    )
    
    # Convert to JSON-safe dict
    safe_dict = metrics.to_json_safe_dict()
    
    # Serialize to JSON
    json_str = json.dumps(safe_dict, indent=2)
    print("   ✅ Successfully serialized TestMetrics with numpy types")
    print(f"   Sample fields:")
    print(f"      decision: {safe_dict['decision']}")
    print(f"      statistic_mean: {safe_dict['statistic_mean']:.4f}")
    print(f"      n_used: {safe_dict['n_used']}/{safe_dict['n_max']}")
    
    # Demo 2: Mock detection
    print("\n2️⃣ Mock Data Detection")
    print("-" * 40)
    
    real_data = {
        "model": "gpt-2",
        "path": "/models/gpt2",
        "score": 0.95
    }
    
    mock_data = {
        "model": "mock_model",
        "path": "/test/dummy_path",
        "score": 0.5
    }
    
    print(f"   Real data valid: {validate_no_mocks(real_data)}")
    print(f"   Mock data valid: {validate_no_mocks(mock_data)}")
    
    # Demo 3: Safe JSON dump with directory creation
    print("\n3️⃣ Safe JSON Dump with Auto Directory Creation")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Nested path that doesn't exist yet
        output_file = Path(tmpdir) / "results" / "session_001" / "metrics.json"
        
        # Safe dump creates directories automatically
        safe_json_dump(safe_dict, output_file, indent=2)
        
        print(f"   ✅ Created nested directories and saved to:")
        print(f"      {output_file.parent.name}/{output_file.name}")
        
        # Verify file exists and can be loaded
        with open(output_file) as f:
            loaded = json.load(f)
        print(f"   ✅ Successfully loaded back {len(loaded)} fields")
    
    # Demo 4: Edge cases
    print("\n4️⃣ Edge Case Handling")
    print("-" * 40)
    
    edge_cases = {
        "numpy_bool": np.bool_(True),
        "numpy_int32": np.int32(42),
        "numpy_float64": np.float64(3.14159),
        "numpy_array": np.array([1, 2, 3]),
        "path_object": Path("/tmp/test"),
        "bytes_data": b"test_bytes",
        "tuple_data": (1, 2, 3),
        "nested": {
            "array_2d": np.array([[1, 2], [3, 4]]),
            "mixed_list": [np.int32(1), 2.0, "three"]
        }
    }
    
    # Serialize with custom encoder
    json_str = json.dumps(edge_cases, cls=NumpyJSONEncoder, indent=2)
    parsed = json.loads(json_str)
    
    print("   Successfully handled edge cases:")
    print(f"      numpy_bool → {type(parsed['numpy_bool']).__name__}")
    print(f"      numpy_int32 → {type(parsed['numpy_int32']).__name__}")
    print(f"      numpy_array → {type(parsed['numpy_array']).__name__}")
    print(f"      path_object → '{parsed['path_object']}'")
    print(f"      bytes_data → '{parsed['bytes_data'][:8]}...'")
    
    # Demo 5: Summary statistics
    print("\n5️⃣ Test Result Summary Format")
    print("-" * 40)
    
    summary = {
        "timestamp": "2024-01-01T12:00:00",
        "decision": metrics.decision,
        "hypothesis": metrics.hypothesis,
        "alpha": metrics.alpha,
        "beta": metrics.beta,
        "n_used": metrics.n_used,
        "n_max": metrics.n_max,
        "statistic_mean": float(metrics.statistic_mean),
        "statistic_var": float(metrics.statistic_var),
        "stopping_time": metrics.stopping_time,
        "time_total_seconds": metrics.t_total,
        "time_per_query_ms": metrics.t_per_query * 1000
    }
    
    print("   Summary format:")
    print(json.dumps(summary, indent=2)[:300] + "...")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete! The system provides:")
    print("   • Proper numpy type conversion for JSON")
    print("   • Mock data detection and separation")
    print("   • Automatic directory creation")
    print("   • Comprehensive metrics logging")
    print("   • No mock contamination in real results")

if __name__ == "__main__":
    main()