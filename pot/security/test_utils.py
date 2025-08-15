"""
Shared test utilities for security module tests
"""

def run_all_tests(test_functions, module_name="Tests"):
    """
    Run all test functions and report results
    
    Args:
        test_functions: List of test functions to run
        module_name: Name of the test module
        
    Returns:
        Number of passed tests
    """
    print(f"\n{'='*60}")
    print(f"{module_name}")
    print('='*60)
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
            print(f"  ✓ {test_func.__name__} passed")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test_func.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test_func.__name__} error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print('='*60)
    
    return passed
