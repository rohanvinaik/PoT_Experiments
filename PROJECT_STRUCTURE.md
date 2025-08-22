# Project Structure Overview

## Clean Repository Organization

The Proof-of-Training (PoT) framework repository is now organized as follows:

### Root Directory (Minimal & Clean)
Contains only essential configuration files:
- `.gitignore` - Git ignore rules
- `Makefile` - Build automation
- `pyproject.toml` - Python project configuration
- `pytest.ini` - Pytest configuration
- `requirements*.txt` - Dependency specifications
- `README.md` - Project overview
- `CLAUDE.md` - Critical execution instructions

### Core Directories

#### `/src/` - Source Code
- `/src/pot/` - Main PoT framework implementation (relocated from `/pot/`)
  - `core/` - Statistical verification & challenges
  - `security/` - Cryptographic protocols
  - `zk/` - Zero-knowledge proof system
  - `lm/` - Language model verification
  - `vision/` - Vision model verification

#### `/pot/` - New Verifier Module
- `/pot/verifier/` - Clean verifier implementation
  - `core/` - Statistical testing and decision logic
  - `lm/` - Model interfaces (HF, API, echo)
  - `logging/` - Audit trail generation

#### `/scripts/` - All Executable Scripts (Organized)
- **Root level**: Core validation scripts
- `/analysis/` - Analysis and debugging tools
- `/attack/` - Attack testing framework
- `/colab/` - Google Colab integration (30+ scripts)
- `/monitoring/` - System monitoring tools
- `/pipeline/` - Pipeline execution scripts
- `/utilities/` - Utility and helper scripts

#### `/tests/` - All Test Files (Centralized)
- Unit and integration tests
- `/archived/` - Legacy test files
- Colab test suites

#### `/docs/` - All Documentation (Organized)
- `/analysis/` - Technical analysis reports
- `/reports/` - Debug and test reports
- `/validation/` - Validation evidence and guides
- `/ablation_plots/` - Visualization plots
- `/api/`, `/guides/`, `/papers/` - Reference documentation

#### `/configs/` - Configuration Files
- YAML configurations for experiments
- ZK system configurations

#### `/manifests/` - Experiment Manifests
- YAML manifests for reproducible experiments
- NeurIPS demo configurations

#### `/data/` - Data Files
- JSON data files
- Test datasets

#### `/experimental_results/` - Test Outputs
- Timestamped test results
- Rolling metrics
- Validation reports

#### Other Directories:
- `/notebooks/` - Jupyter notebooks
- `/examples/` - Usage examples
- `/tools/` - Additional tools
- `/logs/` - Log files
- `/backup/` - Backup files
- `/build/`, `/proofs/`, `/runs/` - Build artifacts and outputs

## Key Improvements Made

1. **Clean Root Directory**: Moved 80+ files from root to appropriate subdirectories
2. **Organized Scripts**: Created logical subdirectories in `/scripts/` by functionality
3. **Centralized Tests**: All test files now in `/tests/` directory
4. **Documentation Structure**: Clear hierarchy in `/docs/` with categorized content
5. **Maintained Functionality**: All paths in CLAUDE.md remain valid and functional

## For Reviewers

The repository is now:
- **Clean**: Root directory contains only essential files
- **Organized**: Clear directory structure with logical grouping
- **Documented**: README files in key directories explain contents
- **Functional**: All scripts maintain their original functionality
- **Reviewer-friendly**: Easy to navigate and understand the codebase structure

Start with:
1. `README.md` - Project overview
2. `CLAUDE.md` - Execution instructions
3. `/scripts/README.md` - Script organization
4. `/docs/README.md` - Documentation structure