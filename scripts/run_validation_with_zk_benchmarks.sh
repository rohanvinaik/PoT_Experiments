#!/bin/bash

# E2E Validation with ZK Benchmarking Wrapper Script
# This script provides an easy interface to run model validation with ZK benchmarking

set -e  # Exit on error

# Default values
REF_MODEL="gpt2"
CAND_MODEL="distilgpt2"
MODE="audit"
CIRCUIT_SIZES="tiny small"
BENCHMARK_TYPES="proof verify"
OUTPUT_DIR="outputs/validation_with_zk"
ENABLE_ZK=false
ENABLE_OPTIMIZATION=false
ENABLE_ROBUSTNESS=false
STORE_ARTIFACTS=false
VERBOSE=false

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run E2E validation pipeline with integrated ZK proof system benchmarking.

OPTIONS:
    -r, --ref-model MODEL       Reference model (default: gpt2)
    -c, --cand-model MODEL      Candidate model (default: distilgpt2)
    -m, --mode MODE            Testing mode: quick/audit/extended (default: audit)
    -s, --circuit-sizes SIZES  Space-separated circuit sizes (default: tiny small)
                               Options: tiny small medium large huge
    -t, --benchmark-types TYPES Space-separated benchmark types (default: proof verify)
                               Options: proof verify memory throughput all
    -o, --output-dir DIR       Output directory (default: outputs/validation_with_zk)
    -z, --enable-zk           Enable ZK proof generation
    -O, --optimize            Run circuit optimization tests
    -R, --robustness          Run adversarial robustness tests  
    -a, --store-artifacts     Store benchmark artifacts for tracking
    -v, --verbose             Enable verbose output
    -h, --help               Display this help message

EXAMPLES:
    # Basic validation with ZK benchmarks
    $0 -r gpt2 -c distilgpt2

    # Comprehensive ZK testing
    $0 -z -O -R -a -s "tiny small medium" -t all

    # Quick mode with minimal benchmarks
    $0 -m quick -s tiny -t proof

    # Full audit with all features
    $0 -m audit -z -O -R -a -v

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--ref-model)
            REF_MODEL="$2"
            shift 2
            ;;
        -c|--cand-model)
            CAND_MODEL="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -s|--circuit-sizes)
            CIRCUIT_SIZES="$2"
            shift 2
            ;;
        -t|--benchmark-types)
            BENCHMARK_TYPES="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -z|--enable-zk)
            ENABLE_ZK=true
            shift
            ;;
        -O|--optimize)
            ENABLE_OPTIMIZATION=true
            shift
            ;;
        -R|--robustness)
            ENABLE_ROBUSTNESS=true
            shift
            ;;
        -a|--store-artifacts)
            STORE_ARTIFACTS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Print configuration
echo "=========================================="
echo "E2E VALIDATION WITH ZK BENCHMARKING"
echo "=========================================="
echo "Reference Model: $REF_MODEL"
echo "Candidate Model: $CAND_MODEL"
echo "Testing Mode: $MODE"
echo "Circuit Sizes: $CIRCUIT_SIZES"
echo "Benchmark Types: $BENCHMARK_TYPES"
echo "Output Directory: $OUTPUT_DIR"
echo "Enable ZK Proofs: $ENABLE_ZK"
echo "Circuit Optimization: $ENABLE_OPTIMIZATION"
echo "Robustness Testing: $ENABLE_ROBUSTNESS"
echo "Store Artifacts: $STORE_ARTIFACTS"
echo "=========================================="
echo

# Build command
CMD="python scripts/run_e2e_validation.py"
CMD="$CMD --ref-model $REF_MODEL"
CMD="$CMD --cand-model $CAND_MODEL"
CMD="$CMD --mode $MODE"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --enable-zk-benchmarks"
CMD="$CMD --zk-circuit-sizes $CIRCUIT_SIZES"
CMD="$CMD --zk-benchmark-types $BENCHMARK_TYPES"

if [ "$ENABLE_ZK" = true ]; then
    CMD="$CMD --enable-zk"
fi

if [ "$ENABLE_OPTIMIZATION" = true ]; then
    CMD="$CMD --zk-optimization-tests"
fi

if [ "$ENABLE_ROBUSTNESS" = true ]; then
    CMD="$CMD --zk-robustness-tests"
fi

if [ "$STORE_ARTIFACTS" = true ]; then
    CMD="$CMD --zk-store-artifacts"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

# Run the command
echo "Executing: $CMD"
echo
eval $CMD
EXIT_CODE=$?

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo
echo "=========================================="
echo "VALIDATION COMPLETED"
echo "=========================================="
echo "Duration: ${MINUTES}m ${SECONDS}s"
echo "Exit Code: $EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR"

if [ "$STORE_ARTIFACTS" = true ]; then
    echo "Artifacts stored in: $OUTPUT_DIR/zk_benchmarks/artifacts"
fi

echo "=========================================="

exit $EXIT_CODE