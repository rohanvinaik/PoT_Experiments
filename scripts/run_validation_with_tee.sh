#!/usr/bin/env bash

# =========================================================
# PoT E2E Validation with TEE Attestation
# =========================================================
# This script runs the complete validation pipeline with
# optional TEE attestation and API security features
# =========================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
REF_MODEL="gpt2"
CAND_MODEL="distilgpt2"
MODE="audit"
TEE_PROVIDER="none"
ENABLE_API_BINDING=""
ATTESTATION_POLICY="moderate"
OUTPUT_DIR="outputs/validation_reports"
DRY_RUN=""
VERBOSE=""
SKIP_ZK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ref-model)
            REF_MODEL="$2"
            shift 2
            ;;
        --cand-model)
            CAND_MODEL="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --tee-provider)
            TEE_PROVIDER="$2"
            shift 2
            ;;
        --enable-api-binding)
            ENABLE_API_BINDING="--enable-api-binding"
            shift
            ;;
        --attestation-policy)
            ATTESTATION_POLICY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --skip-zk)
            SKIP_ZK="--skip-zk"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --ref-model MODEL        Reference model (default: gpt2)"
            echo "  --cand-model MODEL       Candidate model (default: distilgpt2)"
            echo "  --mode MODE              Testing mode: quick, audit, extended (default: audit)"
            echo "  --tee-provider PROVIDER  TEE provider: sgx, sev, nitro, vendor, mock, none (default: none)"
            echo "  --enable-api-binding     Enable API transcript binding"
            echo "  --attestation-policy POL Attestation policy: strict, moderate, relaxed (default: moderate)"
            echo "  --output-dir DIR         Output directory (default: outputs/validation_reports)"
            echo "  --dry-run                Run in dry-run mode"
            echo "  --verbose                Enable verbose output"
            echo "  --skip-zk                Skip ZK proof generation"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic validation"
            echo "  $0"
            echo ""
            echo "  # With SGX attestation"
            echo "  $0 --tee-provider sgx --enable-api-binding"
            echo ""
            echo "  # Quick test with mock TEE"
            echo "  $0 --mode quick --tee-provider mock --dry-run"
            echo ""
            echo "  # Full audit with vendor commitment"
            echo "  $0 --mode extended --tee-provider vendor --attestation-policy strict"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to root directory
cd "$ROOT_DIR"

echo -e "${BLUE}========================================================${NC}"
echo -e "${BLUE}    PoT E2E Validation with TEE Attestation${NC}"
echo -e "${BLUE}========================================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Reference Model:    $REF_MODEL"
echo "  Candidate Model:    $CAND_MODEL"
echo "  Testing Mode:       $MODE"
echo "  TEE Provider:       $TEE_PROVIDER"
echo "  Attestation Policy: $ATTESTATION_POLICY"
echo "  API Binding:        ${ENABLE_API_BINDING:-disabled}"
echo "  Output Directory:   $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python scripts/run_e2e_validation.py"
CMD="$CMD --ref-model $REF_MODEL"
CMD="$CMD --cand-model $CAND_MODEL"
CMD="$CMD --mode $MODE"
CMD="$CMD --tee-provider $TEE_PROVIDER"
CMD="$CMD --attestation-policy $ATTESTATION_POLICY"
CMD="$CMD --output-dir $OUTPUT_DIR"

# Add optional flags
if [[ -n "$ENABLE_API_BINDING" ]]; then
    CMD="$CMD $ENABLE_API_BINDING"
fi

if [[ -n "$DRY_RUN" ]]; then
    CMD="$CMD $DRY_RUN"
    echo -e "${YELLOW}>>> DRY RUN MODE <<<${NC}"
fi

if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi

if [[ -n "$SKIP_ZK" ]]; then
    CMD="$CMD --disable-zk"
fi

# Run validation
echo -e "${GREEN}Starting validation pipeline...${NC}"
echo "Command: $CMD"
echo ""

if $CMD; then
    echo ""
    echo -e "${GREEN}✅ Validation completed successfully!${NC}"
    
    # If TEE was enabled, run additional tests
    if [[ "$TEE_PROVIDER" != "none" ]]; then
        echo ""
        echo -e "${BLUE}Running TEE attestation tests...${NC}"
        python scripts/run_tee_validation.py --provider "$TEE_PROVIDER" --policy "$ATTESTATION_POLICY" || true
    fi
    
    # Show output location
    echo ""
    echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
    
    # List generated files
    echo -e "${YELLOW}Generated files:${NC}"
    ls -la "$OUTPUT_DIR"/*.json 2>/dev/null | tail -5 || echo "  No JSON files generated"
    ls -la "$OUTPUT_DIR"/*.html 2>/dev/null | tail -1 || echo "  No HTML report generated"
else
    echo ""
    echo -e "${RED}❌ Validation failed!${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================================${NC}"
echo -e "${GREEN}All tests completed!${NC}"
echo -e "${BLUE}========================================================${NC}"