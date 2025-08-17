#!/usr/bin/env python3
"""
Fix TailChasing issues identified in the PoT codebase
"""

import os
import sys
from pathlib import Path
import re

def fix_file_references():
    """Fix missing __file__ references in Python files"""
    
    # Files that need fixing (from TailChasingFixer report)
    files_to_fix = [
        'comprehensive_validation.py',
        'experimental_report.py',
        'experimental_report_final.py', 
        'experimental_report_fixed.py',
        'experimental_results/stress_test.py',
        'experimental_results/validation_experiment.py',
        'scripts/run_attack.py',
        'scripts/run_generate_reference.py',
        'scripts/run_grid.py',
        'scripts/run_plots.py',
        'scripts/run_verify.py',
        'scripts/run_baselines.py',
        'scripts/run_coverage.py'
    ]
    
    base_path = Path('/Users/rohanvinaik/PoT_Experiments')
    fixed_count = 0
    
    for file_path in files_to_fix:
        full_path = base_path / file_path
        if not full_path.exists():
            print(f"Warning: {full_path} not found")
            continue
            
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Check if file already has the fix at the end
        if content.strip().endswith('__file__ = None  # TODO: Set appropriate value'):
            # Remove the improper fix
            lines = content.split('\n')
            if lines[-1] == '__file__ = None  # TODO: Set appropriate value':
                lines = lines[:-1]
                content = '\n'.join(lines)
        
        # Fix pattern: sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if '__file__' in content and 'sys.path.insert' in content:
            # Replace the problematic line with a working alternative
            pattern = r'sys\.path\.insert\(0,\s*os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)'
            replacement = 'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0] if __name__ == "__main__" else __file__))))'
            
            new_content = re.sub(pattern, replacement, content)
            
            if new_content != content:
                with open(full_path, 'w') as f:
                    f.write(new_content)
                print(f"Fixed __file__ reference in {file_path}")
                fixed_count += 1
    
    return fixed_count


def remove_duplicate_functions():
    """Remove or consolidate duplicate functions"""
    
    duplicates = [
        # (file, function, duplicate_in)
        ('pot/lm/attacks.py', 'targeted_finetune', 'pot/core/attacks.py'),
        ('pot/lm/attacks.py', 'limited_distillation', 'pot/core/attacks.py'),
    ]
    
    fixes_applied = 0
    
    for file_path, func_name, original_file in duplicates:
        full_path = Path('/Users/rohanvinaik/PoT_Experiments') / file_path
        
        if not full_path.exists():
            continue
            
        # Read the file
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Check if it's already importing from core
        if f'from pot.core.attacks import {func_name}' in content:
            print(f"Already importing {func_name} from core in {file_path}")
            continue
        
        # Replace local definition with import
        # This is a simplified approach - in production would use AST
        if func_name in content:
            lines = content.split('\n')
            new_lines = []
            skip_until_next_def = False
            
            for line in lines:
                if f'def {func_name}' in line:
                    skip_until_next_def = True
                    continue
                if skip_until_next_def and (line.startswith('def ') or line.startswith('class ')):
                    skip_until_next_def = False
                if not skip_until_next_def:
                    new_lines.append(line)
            
            # Add import at the top
            import_added = False
            final_lines = []
            for line in new_lines:
                if not import_added and line.startswith('from pot.core'):
                    final_lines.append(f'from pot.core.attacks import {func_name}')
                    import_added = True
                final_lines.append(line)
            
            if not import_added:
                # Add after other imports
                for i, line in enumerate(final_lines):
                    if line.startswith('import ') or line.startswith('from '):
                        continue
                    final_lines.insert(i, f'from pot.core.attacks import {func_name}')
                    break
            
            new_content = '\n'.join(final_lines)
            
            if new_content != content:
                with open(full_path, 'w') as f:
                    f.write(new_content)
                print(f"Fixed duplicate {func_name} in {file_path}")
                fixes_applied += 1
    
    return fixes_applied


def implement_placeholder_functions():
    """Implement placeholder functions that just have 'pass'"""
    
    placeholders = [
        ('pot/eval/baselines.py', 'lightweight_fingerprint'),
        ('pot/eval/plots.py', 'plot_det_curve'),
        ('pot/vision/probes.py', 'render_sine_grating'),
        ('pot/vision/probes.py', 'render_texture'),
    ]
    
    implementations = {
        'lightweight_fingerprint': '''def lightweight_fingerprint(model_outputs):
    """Generate lightweight fingerprint from model outputs"""
    import hashlib
    import numpy as np
    
    # Convert outputs to normalized form
    if isinstance(model_outputs, (list, tuple)):
        outputs = np.array(model_outputs)
    else:
        outputs = model_outputs
    
    # Compute simple statistics
    mean = float(np.mean(outputs))
    std = float(np.std(outputs))
    
    # Create fingerprint
    fingerprint = {
        "mean": mean,
        "std": std,
        "shape": outputs.shape,
        "hash": hashlib.sha256(str(outputs).encode()).hexdigest()[:16]
    }
    
    return fingerprint''',
    
        'plot_det_curve': '''def plot_det_curve(y_true, y_scores, save_path=None):
    """Plot Detection Error Tradeoff (DET) curve"""
    from sklearn.metrics import det_curve
    import matplotlib.pyplot as plt
    import numpy as np
    
    fpr, fnr, thresholds = det_curve(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, fnr, 'b-', linewidth=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('False Negative Rate')
    ax.set_title('DET Curve')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig''',
    
        'render_sine_grating': '''def render_sine_grating(size=(224, 224), frequency=10, angle=0):
    """Render a sine grating pattern"""
    import numpy as np
    
    h, w = size
    x = np.linspace(0, 2*np.pi*frequency, w)
    y = np.linspace(0, 2*np.pi*frequency, h)
    X, Y = np.meshgrid(x, y)
    
    # Rotate by angle
    angle_rad = np.deg2rad(angle)
    Xr = X * np.cos(angle_rad) - Y * np.sin(angle_rad)
    
    # Generate sine pattern
    grating = (np.sin(Xr) + 1) / 2  # Normalize to [0, 1]
    
    # Convert to RGB
    grating_rgb = np.stack([grating] * 3, axis=-1)
    
    return grating_rgb.astype(np.float32)''',
    
        'render_texture': '''def render_texture(size=(224, 224), texture_type='noise'):
    """Render various texture patterns"""
    import numpy as np
    
    h, w = size
    
    if texture_type == 'noise':
        texture = np.random.rand(h, w, 3)
    elif texture_type == 'checkerboard':
        block_size = 16
        texture = np.zeros((h, w))
        for i in range(0, h, block_size*2):
            for j in range(0, w, block_size*2):
                texture[i:i+block_size, j:j+block_size] = 1
                texture[i+block_size:i+2*block_size, j+block_size:j+2*block_size] = 1
        texture = np.stack([texture] * 3, axis=-1)
    elif texture_type == 'gradient':
        x_grad = np.linspace(0, 1, w)
        y_grad = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x_grad, y_grad)
        texture = np.stack([X, Y, (X+Y)/2], axis=-1)
    else:
        # Default to random noise
        texture = np.random.rand(h, w, 3)
    
    return texture.astype(np.float32)'''
    }
    
    fixes_applied = 0
    base_path = Path('/Users/rohanvinaik/PoT_Experiments')
    
    for file_path, func_name in placeholders:
        full_path = base_path / file_path
        
        if not full_path.exists():
            print(f"Warning: {full_path} not found")
            continue
        
        if func_name not in implementations:
            print(f"No implementation for {func_name}")
            continue
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Check if function just has 'pass'
        pattern = rf'def {func_name}\([^)]*\):\s*(?:"""[^"]*"""\s*)?pass'
        
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            # Replace with implementation
            new_content = re.sub(pattern, implementations[func_name], content)
            
            with open(full_path, 'w') as f:
                f.write(new_content)
            
            print(f"Implemented {func_name} in {file_path}")
            fixes_applied += 1
    
    return fixes_applied


def main():
    """Run all fixes"""
    print("="*60)
    print("TailChasing Issue Fixer for PoT Experiments")
    print("="*60)
    
    total_fixes = 0
    
    print("\n1. Fixing __file__ references...")
    fixes = fix_file_references()
    total_fixes += fixes
    print(f"   Fixed {fixes} __file__ references")
    
    print("\n2. Removing duplicate functions...")
    fixes = remove_duplicate_functions()
    total_fixes += fixes
    print(f"   Fixed {fixes} duplicate functions")
    
    print("\n3. Implementing placeholder functions...")
    fixes = implement_placeholder_functions()
    total_fixes += fixes
    print(f"   Implemented {fixes} placeholder functions")
    
    print("\n" + "="*60)
    print(f"Total fixes applied: {total_fixes}")
    print("="*60)
    
    return total_fixes


if __name__ == "__main__":
    fixes = main()
    sys.exit(0 if fixes > 0 else 1)