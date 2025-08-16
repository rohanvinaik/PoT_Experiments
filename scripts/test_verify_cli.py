#!/usr/bin/env python
"""
Test script for the enhanced verification CLI.
Demonstrates various parameter combinations and features.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


def run_cli_test(name: str, args: list, check_artifacts: bool = True):
    """Run a CLI test with given arguments."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    
    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Add output directory to args
        full_args = ["python", "scripts/run_verify_enhanced.py"] + args + ["--outdir", tmpdir]
        
        print(f"Command: {' '.join(full_args)}")
        
        try:
            # Run the command
            result = subprocess.run(
                full_args,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"Exit code: {result.returncode}")
            
            if result.stdout:
                print("Output:")
                print(result.stdout)
            
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            
            # Check artifacts if requested
            if check_artifacts and result.returncode == 0:
                outdir = Path(tmpdir)
                
                # Check for required artifacts
                artifacts = [
                    "commit.json",
                    "commitment.json",
                    "reveal.json",
                    "decision.json"
                ]
                
                for artifact in artifacts:
                    artifact_path = outdir / artifact
                    if artifact_path.exists():
                        print(f"✓ {artifact} created")
                        
                        # Show decision.json content
                        if artifact == "decision.json":
                            with open(artifact_path) as f:
                                decision = json.load(f)
                            print(f"  Decision: {decision.get('decision')}")
                            print(f"  Stopping time: {decision.get('stopping_time')}")
                            print(f"  Mean: {decision.get('mean', 0):.4f}")
                    else:
                        print(f"✗ {artifact} missing")
                
                # Check for plots
                plots_dir = outdir / "plots"
                if plots_dir.exists():
                    plot_files = list(plots_dir.glob("*.png"))
                    if plot_files:
                        print(f"✓ Generated {len(plot_files)} plots")
                
                # Check for audit files
                audit_files = list(outdir.glob("audit_*.json"))
                if audit_files:
                    print(f"✓ Generated {len(audit_files)} audit record(s)")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("✗ Test timed out")
            return False
        except Exception as e:
            print(f"✗ Test failed: {e}")
            return False


def create_test_config():
    """Create a minimal test configuration file."""
    config = {
        "experiment": "test_verify",
        "models": {
            "reference_path": "tests/fixtures/mock_model.pt",
            "test_path": "tests/fixtures/mock_model.pt"
        },
        "challenges": {
            "families": [
                {
                    "family": "vision:freq",
                    "params": {
                        "freq_range": [0.1, 10.0],
                        "contrast_range": [0.1, 1.0]
                    }
                }
            ]
        },
        "verification": {
            "distances": ["logits_l2"],
            "tau_grid": [0.01, 0.05, 0.1]
        }
    }
    
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config, f)
        return f.name


def create_mock_model():
    """Create a mock model file for testing."""
    import torch
    import tempfile
    
    # Create a simple linear model
    model = torch.nn.Linear(4, 10)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        return f.name


def main():
    """Run all CLI tests."""
    print("Enhanced Verification CLI Test Suite")
    print("="*60)
    
    # Create test fixtures
    print("Creating test fixtures...")
    
    # Create mock model
    model_path = create_mock_model()
    
    # Create config with mock model path
    config = {
        "experiment": "test_verify",
        "models": {
            "reference_path": model_path,
            "test_path": model_path
        },
        "challenges": {
            "families": [
                {
                    "family": "vision:freq",
                    "params": {
                        "freq_range": [0.1, 10.0],
                        "contrast_range": [0.1, 1.0]
                    }
                }
            ]
        },
        "verification": {
            "distances": ["logits_l2"],
            "tau_grid": [0.01, 0.05, 0.1]
        }
    }
    
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    # Test 1: Basic verification with defaults
    success = run_cli_test(
        "Basic verification with defaults",
        ["--config", config_path]
    )
    
    # Test 2: Custom error bounds
    success = run_cli_test(
        "Custom error bounds",
        [
            "--config", config_path,
            "--alpha", "0.05",
            "--beta", "0.10",
            "--tau-id", "0.1"
        ]
    )
    
    # Test 3: With master key and nonce
    success = run_cli_test(
        "With master key and nonce",
        [
            "--config", config_path,
            "--master-key", "0" * 64,
            "--nonce", "test_nonce_123",
            "--n-max", "100"
        ]
    )
    
    # Test 4: Custom challenge parameters
    success = run_cli_test(
        "Custom challenge parameters",
        [
            "--config", config_path,
            "--family", "vision:freq",
            "--params", '{"freq_range": [1.0, 5.0], "contrast_range": [0.2, 0.8]}',
            "--n-max", "50"
        ]
    )
    
    # Test 5: Leakage control parameters
    success = run_cli_test(
        "Leakage control parameters",
        [
            "--config", config_path,
            "--reuse-u", "3",
            "--rho-max", "0.2",
            "--n-max", "30"
        ]
    )
    
    # Test 6: With equivalence transforms
    success = run_cli_test(
        "With equivalence transforms",
        [
            "--config", config_path,
            "--equiv", "rotate90", "flip_horizontal",
            "--wrapper-budget-proxy", "0.15"
        ]
    )
    
    # Test 7: CPU-only mode
    success = run_cli_test(
        "CPU-only mode",
        [
            "--config", config_path,
            "--cpu-only",
            "--n-max", "20"
        ]
    )
    
    # Test 8: Verbose mode with no plots
    success = run_cli_test(
        "Verbose mode without plots",
        [
            "--config", config_path,
            "--verbose",
            "--no-plots",
            "--n-max", "10"
        ]
    )
    
    # Test 9: Different boundary type
    success = run_cli_test(
        "Mixture boundary type",
        [
            "--config", config_path,
            "--boundary", "MB",
            "--n-max", "25"
        ]
    )
    
    # Clean up
    try:
        os.unlink(config_path)
        os.unlink(model_path)
    except:
        pass
    
    print("\n" + "="*60)
    print("CLI Test Suite Complete")
    print("="*60)


if __name__ == "__main__":
    main()