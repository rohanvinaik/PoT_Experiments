#!/usr/bin/env python3
"""
Automatically fix common issues in test files.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict

def add_path_setup(filepath: str) -> bool:
    """Add path setup to a Python file if it imports from pot."""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check if file imports from pot
    imports_pot = any('from pot' in line or 'import pot' in line for line in lines)
    if not imports_pot:
        return False
    
    # Check if already has path setup
    has_path_setup = any('sys.path' in line for line in lines)
    if has_path_setup:
        return False
    
    # Find where to insert path setup (after imports)
    import_section_end = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_section_end = i + 1
        elif import_section_end > 0 and line.strip() and not line.startswith('#'):
            break
    
    # Add path setup
    path_setup = [
        '\n',
        '# Add parent directory to path for pot imports\n',
        'import sys\n',
        'import os\n',
        'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n',
        '\n'
    ]
    
    # Check if sys and os are already imported
    has_sys = any('import sys' in line for line in lines[:import_section_end])
    has_os = any('import os' in line for line in lines[:import_section_end])
    
    if has_sys:
        path_setup.remove('import sys\n')
    if has_os:
        path_setup.remove('import os\n')
    
    # Insert the path setup
    for i, line in enumerate(path_setup):
        lines.insert(import_section_end + i, line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    return True

def fix_incorrect_imports(filepath: str) -> List[str]:
    """Fix incorrect import names in a file."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    fixes = []
    
    # Fix incorrect class/function names
    replacements = [
        ('SequentialVerifier', 'SequentialTester'),
        ('from pot.core.fingerprint import compute_fingerprint', 
         'from pot.core.fingerprint import FingerprintConfig'),
        ('from pot.core.fingerprint import fingerprint_model',
         'from pot.core.fingerprint import FingerprintConfig'),
        ('from pot.core.fingerprint import BehavioralFingerprint',
         'from pot.core.fingerprint import FingerprintConfig'),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            fixes.append(f"Replaced '{old}' with '{new}'")
    
    # Fix challenge families
    family_patterns = [
        (r"family\s*=\s*['\"]numeric['\"]", 'family="vision:freq"'),
        (r"family\s*:\s*['\"]numeric['\"]", 'family: "vision:freq"'),
    ]
    
    for pattern, replacement in family_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            fixes.append(f"Fixed challenge family from 'numeric' to 'vision:freq'")
    
    # Fix parameter names
    param_patterns = [
        (r'SequentialTester\([^)]*tau\s*=', 'SequentialTester(tau0=0.4, tau1=0.6, '),
        (r'ChallengeConfig\([^)]*num_challenges\s*=', 'ChallengeConfig(n='),
    ]
    
    for pattern, issue in param_patterns:
        if re.search(pattern, content):
            fixes.append(f"Found parameter issue: {issue}")
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
    
    return fixes

def fix_wrapper_detection_exports(filepath: str) -> bool:
    """Fix wrapper detection export issues."""
    
    if 'wrapper_detection.py' not in str(filepath):
        return False
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check if detect_wrapper is defined
    has_detect_wrapper = any('def detect_wrapper' in line for line in lines)
    
    if has_detect_wrapper:
        # Make sure it's exported
        has_all = any('__all__' in line for line in lines)
        
        if not has_all:
            # Add __all__ at the top of the file after imports
            import_end = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_end = i + 1
            
            lines.insert(import_end + 1, '\n__all__ = ["detect_wrapper"]\n\n')
            
            with open(filepath, 'w') as f:
                f.writelines(lines)
            return True
    
    return False

def main():
    """Main entry point."""
    print("🔧 PoT Framework Test Issue Fixer")
    print("=" * 70)
    
    # Load issues report
    if not os.path.exists('test_issues_report.json'):
        print("❌ No test_issues_report.json found. Run check_all_tests.py first.")
        return 1
    
    with open('test_issues_report.json', 'r') as f:
        issues = json.load(f)
    
    print(f"Found {len(issues)} files with issues to fix\n")
    
    fixed_files = []
    skipped_files = []
    
    for filepath, file_issues in issues.items():
        print(f"\n📄 Processing: {filepath}")
        print("-" * 40)
        
        fixes_made = []
        
        # Skip the checker script itself
        if 'check_all_tests.py' in filepath:
            print("  → Skipping checker script itself")
            skipped_files.append(filepath)
            continue
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"  → File not found, skipping")
            skipped_files.append(filepath)
            continue
        
        # Fix path setup issues
        if any('path setup' in issue for issue in file_issues):
            if add_path_setup(filepath):
                fixes_made.append("Added path setup")
                print("  ✅ Added path setup for pot imports")
        
        # Fix incorrect imports
        import_fixes = fix_incorrect_imports(filepath)
        if import_fixes:
            fixes_made.extend(import_fixes)
            for fix in import_fixes:
                print(f"  ✅ {fix}")
        
        # Fix wrapper detection exports
        if any('detect_wrapper' in issue for issue in file_issues):
            if fix_wrapper_detection_exports(filepath):
                fixes_made.append("Fixed wrapper detection exports")
                print("  ✅ Fixed wrapper detection exports")
        
        if fixes_made:
            fixed_files.append((filepath, fixes_made))
        else:
            print("  ℹ️ No automatic fixes applied")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Files processed: {len(issues)}")
    print(f"  Files fixed: {len(fixed_files)}")
    print(f"  Files skipped: {len(skipped_files)}")
    
    if fixed_files:
        print("\n✅ FIXED FILES:")
        for filepath, fixes in fixed_files[:10]:  # Show first 10
            print(f"  {filepath}")
            for fix in fixes:
                print(f"    - {fix}")
    
    # Save fix report
    fix_report = {
        'fixed_files': dict(fixed_files),
        'skipped_files': skipped_files,
        'summary': {
            'total_issues': len(issues),
            'files_fixed': len(fixed_files),
            'files_skipped': len(skipped_files)
        }
    }
    
    with open('test_fixes_report.json', 'w') as f:
        json.dump(fix_report, f, indent=2)
    
    print(f"\n📄 Fix report saved to: test_fixes_report.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())