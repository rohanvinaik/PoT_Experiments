#!/usr/bin/env python3
"""
Run all fingerprint-related tests and display summary.
"""

import sys
import subprocess
import time


def run_test_file(test_file: str, description: str) -> bool:
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr and "OK" not in result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        
        if success:
            print(f"✅ PASSED in {elapsed:.2f}s")
        else:
            print(f"❌ FAILED in {elapsed:.2f}s")
            if result.stderr:
                print("Error details:", result.stderr[-500:])  # Last 500 chars
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT after 30s")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run all fingerprint tests."""
    print("="*60)
    print("FINGERPRINT SYSTEM TEST SUITE")
    print("="*60)
    
    tests = [
        ("pot/core/test_fingerprint.py", "Core Fingerprint Unit Tests"),
        ("test_fingerprint_comparison.py", "Fingerprint Comparison Utilities"),
        ("test_vision_fingerprint_integration.py", "Vision Model Integration"),
        ("test_lm_fingerprint_integration.py", "Language Model Integration"),
    ]
    
    results = []
    total_start = time.time()
    
    for test_file, description in tests:
        success = run_test_file(test_file, description)
        results.append((test_file, description, success))
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, _, success in results if success)
    total = len(results)
    
    for test_file, description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {description}")
    
    print("-"*60)
    print(f"Results: {passed}/{total} passed")
    print(f"Total time: {total_elapsed:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())