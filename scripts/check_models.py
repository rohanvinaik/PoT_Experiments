#!/usr/bin/env python3
"""
Quick model availability checker for PoT experiments.
"""
import sys
import os

def check_models():
    """Check for model availability."""
    print("📦 Model Availability Report:")
    
    # Simple check for model setup module
    try:
        from pot.experiments.model_setup import MinimalModelSetup
        setup = MinimalModelSetup()
        available = setup.list_available_models()
        
        all_available = True
        for model_type, presets in available.items():
            print(f"\n{model_type.upper()} Models:")
            for preset, info in presets.items():
                status = '✅' if info.get('available', False) else '❌'
                description = info.get('description', 'N/A')
                print(f"  {status} {preset}: {description}")
                if not info.get('available', False):
                    all_available = False
        
        if all_available:
            print('\n✅ All models available')
        else:
            print('\n⚠️  Some models unavailable - will use fallbacks')
            
    except ImportError as e:
        print(f"\n❌ Cannot check models: {e}")
        print("📋 PoT package may not be properly installed")
        # Don't exit with error since this is expected during setup
        
    except Exception as e:
        print(f"\n⚠️  Model check failed: {e}")
        print("📋 Will use fallback models during experiments")

if __name__ == '__main__':
    check_models()