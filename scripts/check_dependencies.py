#!/usr/bin/env python3
"""
Quick dependency checker for PoT experiments.
"""
import sys

def check_dependencies():
    """Check for required dependencies."""
    missing = []
    dependencies = ['numpy', 'torch', 'scipy', 'pot']
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f'✅ {dep}')
        except ImportError:
            missing.append(dep)
            print(f'❌ {dep}')
    
    if missing:
        print(f'\n🔴 Missing dependencies: {missing}')
        print('📋 Run: make install-deps')
        sys.exit(1)
    else:
        print('\n✅ All dependencies satisfied')

if __name__ == '__main__':
    check_dependencies()