# Scripts Directory Organization

This directory contains all executable scripts for the Proof-of-Training (PoT) framework, organized by functionality.

## Directory Structure

### Core Validation Scripts (Root Level)
- `run_all.sh` - Main validation pipeline runner
- `run_enhanced_diff_test.py` - Enhanced statistical verification with calibration
- `run_statistical_verification.py` - Statistical identity verification
- `test_llm_verification.py` - Language model verification
- `run_fuzzy_verification.py` - Fuzzy hash testing
- `run_provenance_verification.py` - Merkle tree provenance
- `run_zk_validation.py` - Zero-knowledge proof validation
- `run_security_tests_simple.py` - Security verification suite
- `experimental_report_clean.py` - Clean reporting format
- `run_diff.py` - Main differential testing runner
- `run_pipeline_with_models.py` - Model loading pipeline with auto-discovery
- `test_size_fraud_detection.py` - Size fraud detection tests
- `benchmark_*.py` - Performance benchmarking scripts

### `/analysis/` - Analysis and Debugging Tools
- `analyze_tailchasing.py` - Analyze circular dependency issues
- `debug_scorer.py` - Debug scoring functionality
- `find_results.py` - Find and analyze test results

### `/attack/` - Attack Testing Framework
- `pot_attack.sh` - Main attack wrapper script
- Attack simulation and resistance testing tools

### `/colab/` - Google Colab Integration
- Complete suite of Colab-compatible runners
- `COLAB_FINAL.py` - Production Colab runner
- `colab_run_all.py` - Complete test suite for Colab
- Various specialized Colab test scripts

### `/monitoring/` - System Monitoring Tools
- `monitor_and_run.sh` - Monitor and run pipeline
- `monitor_download.py` - Monitor model downloads
- `monitor_pipeline.py` - Pipeline execution monitoring
- `show_progress.sh` - Display progress information
- `speed_monitor.sh` - Monitor execution speed
- `track_*.py` - Model download tracking scripts
- `watchdog.sh` - Process monitoring watchdog

### `/pipeline/` - Pipeline Execution Scripts
- `run_full_pipeline.sh` - Complete pipeline runner for large models
- `run_pot_colab_correct.py` - Corrected Colab pipeline
- `run_paper.sh` - Paper reproduction pipeline

### `/utilities/` - Utility Scripts
- `check_progress.sh` - Check execution progress
- `prepare_*.sh` - Model preparation scripts
- `package_for_colab.sh` - Package for Colab deployment
- `restart_download.py` - Restart interrupted downloads
- `upload_minimal.sh` - Minimal upload script

## Quick Start

For full validation (15-20 minutes):
```bash
bash scripts/run_all.sh
```

For quick testing without ZK proofs (5-10 minutes):
```bash
bash scripts/run_all.sh --skip-zk
```

For model pipeline testing:
```bash
python scripts/run_pipeline_with_models.py --models-dir /Users/rohanvinaik/LLM_Models
```

For size fraud detection:
```bash
python scripts/test_size_fraud_detection.py
```

## Important Notes

- Always run scripts from the repository root: `/Users/rohanvinaik/PoT_Experiments`
- Ensure Python environment has required packages: torch, transformers, numpy, scipy
- Models should be in `/Users/rohanvinaik/LLM_Models/`
- Check CLAUDE.md for detailed execution requirements

