#!/bin/bash
# PoT Attack CLI Wrapper Script
# Provides easy command-line access to attack evaluation tools

# Set Python path to repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}"

# Default Python interpreter
PYTHON=${PYTHON:-python3}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${2}${1}${NC}"
}

# Function to check dependencies
check_dependencies() {
    if ! command -v $PYTHON &> /dev/null; then
        print_color "Error: Python not found. Please install Python 3.8+" "$RED"
        exit 1
    fi

    if ! $PYTHON -c "import torch" 2>/dev/null; then
        print_color "Warning: PyTorch not installed. Some features may not work." "$YELLOW"
    fi

    if ! $PYTHON -c "import click" 2>/dev/null; then
        print_color "Error: Click not installed. Run: pip install click" "$RED"
        exit 1
    fi
}

# Function to show help
show_help() {
    cat << EOF
PoT Attack Evaluation CLI

Usage: pot_attack.sh [OPTIONS] COMMAND [ARGS]...

Commands:
  run-attacks      Run attack suite against model
  generate-report  Generate HTML report from results
  detect-wrapper   Detect if model is wrapped
  benchmark       Run standardized benchmark
  verify          Verify model with PoT
  dashboard       Launch interactive dashboard

Quick Examples:
  ./pot_attack.sh run-attacks -m model.pth -s standard
  ./pot_attack.sh benchmark -c config.yaml -m model.pth
  ./pot_attack.sh dashboard -r results/

Options:
  --help          Show this help message
  --version       Show version
  --verbose       Enable verbose output
  --debug         Enable debug mode

Environment Variables:
  CUDA_VISIBLE_DEVICES  Set GPU devices (default: 0)
  POT_CONFIG_DIR        Configuration directory
  POT_RESULTS_DIR       Results directory

For detailed help on any command:
  ./pot_attack.sh COMMAND --help

EOF
}

# Function to show version
show_version() {
    echo "PoT Attack CLI v1.0.0"
    $PYTHON --version
    $PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: Not installed"
}

# Main script logic
case "$1" in
    --help|-h|"")
        show_help
        ;;
    --version|-v)
        show_version
        ;;
    *)
        # Check dependencies before running
        check_dependencies

        # Run the Python CLI
        $PYTHON -m pot.cli.attack_cli "$@"
        ;;
esac

