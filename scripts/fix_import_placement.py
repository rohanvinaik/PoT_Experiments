#!/usr/bin/env python3
"""
Fix incorrect placement of path setup in import statements.
"""

import os
import re
from pathlib import Path

def fix_import_placement(filepath):
    """Fix path setup that was incorrectly placed inside import statements."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find misplaced path setup inside import statements
    pattern = r'from\s+\w+[.\w]*\s+import\s+\([^)]*\n\n# Add parent directory.*?\n.*?sys\.path\.insert.*?\n\n.*?\)'
    
    if re.search(pattern, content, re.DOTALL):
        print(f"Fixing {filepath}")
        
        # Extract the path setup
        path_setup = """
# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
        
        # Remove misplaced path setup
        content = re.sub(r'\n\n# Add parent directory.*?sys\.path\.insert.*?\n\n', '\n', content, flags=re.DOTALL)
        
        # Find right place to insert (before first pot import)
        lines = content.split('\n')
        insert_idx = -1
        
        for i, line in enumerate(lines):
            if 'from pot' in line or 'import pot' in line:
                insert_idx = i
                break
        
        if insert_idx > 0:
            # Check if sys/os already imported
            has_sys = any('import sys' in line for line in lines[:insert_idx])
            has_os = any('import os' in line for line in lines[:insert_idx])
            
            setup_lines = path_setup.strip().split('\n')
            if has_sys:
                setup_lines = [l for l in setup_lines if 'import sys' not in l]
            if has_os:
                setup_lines = [l for l in setup_lines if 'import os' not in l]
            
            # Insert before the pot import
            for j, setup_line in enumerate(reversed(setup_lines)):
                lines.insert(insert_idx, setup_line)
            lines.insert(insert_idx, '')
            
            content = '\n'.join(lines)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            return True
    
    return False

def main():
    """Fix all files with misplaced imports."""
    
    # Find all Python files that might have the issue
    test_files = []
    for pattern in ['pot/**/*.py', 'tests/**/*.py', 'examples/**/*.py']:
        test_files.extend(Path('.').glob(pattern))
    
    fixed_count = 0
    
    for filepath in test_files:
        if fix_import_placement(str(filepath)):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files with misplaced path setup")

if __name__ == "__main__":
    main()