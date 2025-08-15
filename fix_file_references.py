#!/usr/bin/env python3
"""
Fix all __file__ references in the codebase.
Replaces sys.path manipulation using __file__ with proper pathlib approach.
"""

import os
import re
from pathlib import Path

def fix_file_reference(filepath):
    """Fix __file__ reference in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to match the problematic __file__ usage
    pattern = r'sys\.path\.insert\(0, os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)'
    
    # Check if the pattern exists
    if re.search(pattern, content):
        # Replace with a more robust approach
        new_code = """# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == 'scripts' else Path.cwd()))"""
        
        content = re.sub(pattern, new_code, content)
        
        # Also fix the import section if needed
        if 'import sys' not in content:
            content = 'import sys\n' + content
        if 'from pathlib import Path' not in content and 'Path.cwd()' in content:
            content = content.replace('import sys', 'import sys\nfrom pathlib import Path')
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Fixed: {filepath}")
        return True
    return False

def main():
    """Fix all files with __file__ issues."""
    files_to_fix = [
        '/Users/rohanvinaik/PoT_Experiments/comprehensive_validation.py',
        '/Users/rohanvinaik/PoT_Experiments/experimental_report.py',
        '/Users/rohanvinaik/PoT_Experiments/experimental_report_final.py',
        '/Users/rohanvinaik/PoT_Experiments/experimental_report_fixed.py',
        '/Users/rohanvinaik/PoT_Experiments/experimental_results/stress_test.py',
        '/Users/rohanvinaik/PoT_Experiments/experimental_results/validation_experiment.py',
        '/Users/rohanvinaik/PoT_Experiments/scripts/run_attack.py',
        '/Users/rohanvinaik/PoT_Experiments/scripts/run_generate_reference.py',
        '/Users/rohanvinaik/PoT_Experiments/scripts/run_grid.py',
        '/Users/rohanvinaik/PoT_Experiments/scripts/run_plots.py',
        '/Users/rohanvinaik/PoT_Experiments/scripts/run_verify.py',
    ]
    
    fixed_count = 0
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_file_reference(filepath):
                fixed_count += 1
        else:
            print(f"File not found: {filepath}")
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()