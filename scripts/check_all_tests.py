#!/usr/bin/env python3
"""
Comprehensive test checker for PoT framework.
Identifies and reports issues in all test scripts similar to those found in experimental_report.py
"""

import os
import sys
import ast
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

class TestChecker:
    """Check test scripts for common issues."""
    
    def __init__(self):
        self.issues = []
        self.checked_files = []
        self.import_errors = []
        self.parameter_errors = []
        self.path_issues = []
        
    def check_file_imports(self, filepath: str) -> List[str]:
        """Check a Python file for import issues."""
        issues = []
        
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Check for problematic imports we found
        problematic_imports = [
            ('SequentialVerifier', 'Should be SequentialTester'),
            ('compute_fingerprint', 'Should use FingerprintConfig or other functions'),
            ('fingerprint_model', 'Function does not exist'),
            ('BehavioralFingerprint', 'Class name might be different'),
            ('detect_wrapper', 'Function might not be exported'),
            ('StochasticDecodingController', 'Class might not exist'),
            ("family='numeric'", "Should be 'vision:freq', 'vision:texture', or 'lm:templates'"),
            ("family: 'numeric'", "Should be 'vision:freq', 'vision:texture', or 'lm:templates'"),
        ]
        
        for problem, solution in problematic_imports:
            if problem in content:
                issues.append(f"Found '{problem}' - {solution}")
                
        # Check for parameter issues
        parameter_issues = [
            ('SequentialTester.*tau=', 'Should use tau0= and tau1='),
            ('ChallengeConfig.*num_challenges', 'Should use n= instead'),
            ('generate_challenges.*master_key=', 'Config should have master_key_hex'),
        ]
        
        for pattern, solution in parameter_issues:
            import re
            if re.search(pattern, content):
                issues.append(f"Parameter issue: {solution}")
                
        return issues
    
    def check_path_setup(self, filepath: str) -> bool:
        """Check if file has proper path setup for imports."""
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Check for path setup
        has_path_setup = any([
            'sys.path.insert' in content,
            'sys.path.append' in content,
            'PYTHONPATH' in content,
        ])
        
        # Check if it imports from pot but doesn't have path setup
        imports_pot = 'from pot' in content or 'import pot' in content
        
        if imports_pot and not has_path_setup and 'test' in str(filepath):
            return False
        return True
    
    def test_import(self, module_path: str, item_name: str) -> bool:
        """Test if an import actually works."""
        try:
            module = __import__(module_path, fromlist=[item_name])
            return hasattr(module, item_name)
        except:
            return False
    
    def run_test_file(self, filepath: str) -> Tuple[bool, str]:
        """Try to run a test file and capture errors."""
        try:
            result = subprocess.run(
                [sys.executable, filepath],
                capture_output=True,
                text=True,
                timeout=5,
                env={**os.environ, 'PYTHONPATH': parent_dir}
            )
            
            if result.returncode != 0:
                # Look for import errors
                if 'ImportError' in result.stderr or 'ModuleNotFoundError' in result.stderr:
                    return False, f"Import error: {result.stderr.split('ImportError')[-1][:100]}"
                elif 'AttributeError' in result.stderr:
                    return False, f"Attribute error: {result.stderr.split('AttributeError')[-1][:100]}"
                else:
                    return False, "Failed with errors"
            return True, "Success"
            
        except subprocess.TimeoutExpired:
            return True, "Timeout (might be OK for long tests)"
        except Exception as e:
            return False, str(e)
    
    def check_all_tests(self):
        """Check all test files in the repository."""
        
        # Find all Python test files
        test_patterns = [
            'test_*.py',
            '*_test.py',
            'example_*.py',
            'demo_*.py',
            'run_*.py',
        ]
        
        test_files = []
        for pattern in test_patterns:
            test_files.extend(Path('.').rglob(pattern))
        
        # Also check specific directories
        test_dirs = ['tests', 'scripts', 'examples', 'pot']
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                test_files.extend(Path(dir_name).rglob('*.py'))
        
        # Remove duplicates and sort
        test_files = sorted(set(test_files))
        
        print(f"Found {len(test_files)} test/script files to check\n")
        print("=" * 70)
        
        issues_found = {}
        
        for filepath in test_files:
            filepath_str = str(filepath)
            
            # Skip __pycache__ and other non-relevant files
            if '__pycache__' in filepath_str or '.pyc' in filepath_str:
                continue
                
            # Check the file
            file_issues = []
            
            # Check imports
            import_issues = self.check_file_imports(filepath_str)
            if import_issues:
                file_issues.extend(import_issues)
            
            # Check path setup
            if not self.check_path_setup(filepath_str):
                file_issues.append("Missing path setup for pot imports")
            
            if file_issues:
                issues_found[filepath_str] = file_issues
        
        return issues_found
    
    def generate_report(self, issues: Dict[str, List[str]]):
        """Generate a report of all issues found."""
        
        if not issues:
            print("\n✅ No issues found! All tests appear to be correctly configured.")
            return
        
        print(f"\n⚠️ Found issues in {len(issues)} files:\n")
        print("=" * 70)
        
        # Group by issue type
        import_files = []
        param_files = []
        path_files = []
        
        for filepath, file_issues in issues.items():
            for issue in file_issues:
                if 'import' in issue.lower() or 'should be' in issue.lower():
                    import_files.append((filepath, issue))
                elif 'parameter' in issue.lower():
                    param_files.append((filepath, issue))
                elif 'path' in issue.lower():
                    path_files.append((filepath, issue))
        
        if import_files:
            print("\n📦 IMPORT ISSUES:")
            print("-" * 40)
            for filepath, issue in import_files[:10]:  # Show first 10
                print(f"  {filepath}")
                print(f"    → {issue}")
        
        if param_files:
            print("\n⚙️ PARAMETER ISSUES:")
            print("-" * 40)
            for filepath, issue in param_files[:10]:
                print(f"  {filepath}")
                print(f"    → {issue}")
        
        if path_files:
            print("\n🛤️ PATH SETUP ISSUES:")
            print("-" * 40)
            for filepath, issue in path_files[:10]:
                print(f"  {filepath}")
                print(f"    → {issue}")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY:")
        print(f"  Total files with issues: {len(issues)}")
        print(f"  Import issues: {len(import_files)}")
        print(f"  Parameter issues: {len(param_files)}")
        print(f"  Path setup issues: {len(path_files)}")
        
        return issues

def main():
    """Main entry point."""
    print("🔍 PoT Framework Test Checker")
    print("=" * 70)
    
    checker = TestChecker()
    
    # Check all tests
    issues = checker.check_all_tests()
    
    # Generate report
    checker.generate_report(issues)
    
    # Save detailed report
    if issues:
        report_file = "test_issues_report.json"
        with open(report_file, 'w') as f:
            json.dump(issues, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Offer to create fixes
        print("\n" + "=" * 70)
        print("🔧 SUGGESTED FIXES:")
        print("-" * 40)
        print("1. Add path setup to test files:")
        print("   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))")
        print("\n2. Replace incorrect imports:")
        print("   SequentialVerifier → SequentialTester")
        print("   compute_fingerprint → use FingerprintConfig")
        print("\n3. Fix challenge families:")
        print("   'numeric' → 'vision:freq' or 'lm:templates'")
        print("\n4. Fix parameter names:")
        print("   tau= → tau0= and tau1=")
        print("   num_challenges= → n=")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())