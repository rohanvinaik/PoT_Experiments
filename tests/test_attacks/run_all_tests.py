#!/usr/bin/env python3
"""
Run all attack and defense tests.
"""

import sys
import unittest
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_all_tests():
    """Run all test suites."""
    
    # Discover and load all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        
        if result.failures:
            print("\nFailed tests:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nTests with errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)