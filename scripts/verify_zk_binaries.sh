#!/bin/bash
"""
ZK Binary Verification Script

Comprehensive testing of ZK prover and verifier binaries to ensure
they are functional and properly configured.
"""

set -e  # Exit on any error

# Configuration
PROVER_DIR_DEBUG="pot/zk/prover_halo2/target/debug"
PROVER_DIR_RELEASE="pot/zk/prover_halo2/target/release"
TEST_DIR="/tmp/zk_binary_tests"
LOG_FILE="/tmp/zk_binary_verification.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize
mkdir -p "$TEST_DIR"
echo "ZK Binary Verification Log - $(date)" > "$LOG_FILE"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    echo "$1"
}

# Check if binary exists and is executable
check_binary() {
    local binary_path="$1"
    local binary_name="$2"
    
    if [ -f "$binary_path" ]; then
        if [ -x "$binary_path" ]; then
            log "✅ $binary_name found and executable at $binary_path"
            return 0
        else
            log "⚠️  $binary_name found but not executable at $binary_path"
            return 1
        fi
    else
        log "❌ $binary_name not found at $binary_path"
        return 1
    fi
}

# Get binary version if available
get_binary_version() {
    local binary_path="$1"
    local binary_name="$2"
    
    if [ -x "$binary_path" ]; then
        # Try common version flags
        for flag in "--version" "-v" "--help"; do
            if timeout 5s "$binary_path" "$flag" 2>/dev/null | head -1 > "$TEST_DIR/${binary_name}_version.txt"; then
                local version=$(cat "$TEST_DIR/${binary_name}_version.txt")
                if [ -n "$version" ]; then
                    log "📋 $binary_name version: $version"
                    return 0
                fi
            fi
        done
        log "⚠️  Could not determine $binary_name version"
        return 1
    else
        log "❌ Cannot get version - $binary_name not executable"
        return 1
    fi
}

# Test SGD prover with sample data
test_sgd_prover() {
    local prover_path="$1"
    log "🧪 Testing SGD prover at $prover_path"
    
    # Create minimal test input for SGD prover
    local test_input='{
        "statement": {
            "W_t_root": "'$(printf "%0*d" 64 0)'",
            "W_t1_root": "'$(printf "%0*d" 64 1)'",
            "batch_root": "'$(printf "%0*d" 64 2)'",
            "hparams_hash": "'$(printf "%0*d" 32 3)'",
            "step_nonce": 0,
            "step_number": 1,
            "epoch": 1
        },
        "witness": {
            "weights_before": [0.1, 0.2, 0.3, 0.4],
            "weights_after": [0.11, 0.21, 0.31, 0.41],
            "batch_inputs": [1.0, 2.0, 3.0, 4.0],
            "batch_targets": [0.5, 1.5],
            "learning_rate": 0.01
        }
    }'
    
    local proof_file="$TEST_DIR/sgd_test_proof.bin"
    
    # Test proof generation
    if echo "$test_input" | timeout 30s "$prover_path" > "$proof_file" 2>"$TEST_DIR/sgd_prover_error.log"; then
        if [ -s "$proof_file" ]; then
            local proof_size=$(wc -c < "$proof_file")
            log "✅ SGD prover generated proof ($proof_size bytes)"
            echo "$proof_size" > "$TEST_DIR/sgd_proof_size.txt"
            return 0
        else
            log "❌ SGD prover ran but generated empty proof"
            cat "$TEST_DIR/sgd_prover_error.log" >> "$LOG_FILE"
            return 1
        fi
    else
        log "❌ SGD prover failed to run"
        if [ -f "$TEST_DIR/sgd_prover_error.log" ]; then
            cat "$TEST_DIR/sgd_prover_error.log" >> "$LOG_FILE"
        fi
        return 1
    fi
}

# Test SGD verifier with generated proof
test_sgd_verifier() {
    local verifier_path="$1"
    local proof_file="$TEST_DIR/sgd_test_proof.bin"
    
    log "🔍 Testing SGD verifier at $verifier_path"
    
    if [ ! -f "$proof_file" ]; then
        log "❌ No proof file available for SGD verification test"
        return 1
    fi
    
    # Create verification input
    local verify_input='{
        "statement": {
            "W_t_root": "'$(printf "%0*d" 64 0)'",
            "W_t1_root": "'$(printf "%0*d" 64 1)'",
            "batch_root": "'$(printf "%0*d" 64 2)'",
            "hparams_hash": "'$(printf "%0*d" 32 3)'",
            "step_nonce": 0,
            "step_number": 1,
            "epoch": 1
        },
        "proof": "'$(base64 < "$proof_file" | tr -d '\n')'"
    }'
    
    if echo "$verify_input" | timeout 30s "$verifier_path" > "$TEST_DIR/sgd_verify_result.txt" 2>"$TEST_DIR/sgd_verifier_error.log"; then
        local result=$(cat "$TEST_DIR/sgd_verify_result.txt")
        log "✅ SGD verifier completed: $result"
        return 0
    else
        log "❌ SGD verifier failed"
        if [ -f "$TEST_DIR/sgd_verifier_error.log" ]; then
            cat "$TEST_DIR/sgd_verifier_error.log" >> "$LOG_FILE"
        fi
        return 1
    fi
}

# Test LoRA prover
test_lora_prover() {
    local prover_path="$1"
    log "🧪 Testing LoRA prover at $prover_path"
    
    # Create minimal test input for LoRA prover
    local test_input='{
        "statement": {
            "base_weights_root": "'$(printf "%0*d" 64 0)'",
            "adapter_a_root_before": "'$(printf "%0*d" 32 1)'",
            "adapter_b_root_before": "'$(printf "%0*d" 32 2)'", 
            "adapter_a_root_after": "'$(printf "%0*d" 32 3)'",
            "adapter_b_root_after": "'$(printf "%0*d" 32 4)'",
            "batch_root": "'$(printf "%0*d" 48 5)'",
            "hparams_hash": "'$(printf "%0*d" 32 6)'",
            "rank": 8,
            "alpha": 16.0,
            "step_number": 1,
            "epoch": 1
        },
        "witness": {
            "adapter_a_before": [0.01, 0.02, 0.03, 0.04],
            "adapter_b_before": [0.05, 0.06, 0.07, 0.08],
            "adapter_a_after": [0.011, 0.021, 0.031, 0.041],
            "adapter_b_after": [0.051, 0.061, 0.071, 0.081],
            "adapter_a_gradients": [0.001, 0.001, 0.001, 0.001],
            "adapter_b_gradients": [0.001, 0.001, 0.001, 0.001],
            "batch_inputs": [1.0, 2.0],
            "batch_targets": [0.5, 1.5],
            "learning_rate": 0.001
        }
    }'
    
    local proof_file="$TEST_DIR/lora_test_proof.bin"
    
    # Test proof generation
    if echo "$test_input" | timeout 30s "$prover_path" > "$proof_file" 2>"$TEST_DIR/lora_prover_error.log"; then
        if [ -s "$proof_file" ]; then
            local proof_size=$(wc -c < "$proof_file")
            log "✅ LoRA prover generated proof ($proof_size bytes)"
            echo "$proof_size" > "$TEST_DIR/lora_proof_size.txt"
            return 0
        else
            log "❌ LoRA prover ran but generated empty proof"
            cat "$TEST_DIR/lora_prover_error.log" >> "$LOG_FILE"
            return 1
        fi
    else
        log "❌ LoRA prover failed to run"
        if [ -f "$TEST_DIR/lora_prover_error.log" ]; then
            cat "$TEST_DIR/lora_prover_error.log" >> "$LOG_FILE"
        fi
        return 1
    fi
}

# Test LoRA verifier
test_lora_verifier() {
    local verifier_path="$1"
    local proof_file="$TEST_DIR/lora_test_proof.bin"
    
    log "🔍 Testing LoRA verifier at $verifier_path"
    
    if [ ! -f "$proof_file" ]; then
        log "❌ No proof file available for LoRA verification test"
        return 1
    fi
    
    # Create verification input
    local verify_input='{
        "statement": {
            "base_weights_root": "'$(printf "%0*d" 64 0)'",
            "adapter_a_root_before": "'$(printf "%0*d" 32 1)'",
            "adapter_b_root_before": "'$(printf "%0*d" 32 2)'",
            "adapter_a_root_after": "'$(printf "%0*d" 32 3)'",
            "adapter_b_root_after": "'$(printf "%0*d" 32 4)'",
            "batch_root": "'$(printf "%0*d" 48 5)'",
            "hparams_hash": "'$(printf "%0*d" 32 6)'",
            "rank": 8,
            "alpha": 16.0,
            "step_number": 1,
            "epoch": 1
        },
        "proof": "'$(base64 < "$proof_file" | tr -d '\n')'"
    }'
    
    if echo "$verify_input" | timeout 30s "$verifier_path" > "$TEST_DIR/lora_verify_result.txt" 2>"$TEST_DIR/lora_verifier_error.log"; then
        local result=$(cat "$TEST_DIR/lora_verify_result.txt")
        log "✅ LoRA verifier completed: $result"
        return 0
    else
        log "❌ LoRA verifier failed"
        if [ -f "$TEST_DIR/lora_verifier_error.log" ]; then
            cat "$TEST_DIR/lora_verifier_error.log" >> "$LOG_FILE"
        fi
        return 1
    fi
}

# Check Rust environment
check_rust_environment() {
    log "🦀 Checking Rust environment..."
    
    if command -v rustc >/dev/null 2>&1; then
        local rust_version=$(rustc --version)
        log "✅ Rust compiler: $rust_version"
    else
        log "❌ Rust compiler not found"
        return 1
    fi
    
    if command -v cargo >/dev/null 2>&1; then
        local cargo_version=$(cargo --version)
        log "✅ Cargo: $cargo_version"
    else
        log "❌ Cargo not found"
        return 1
    fi
    
    return 0
}

# Build binaries if they don't exist
build_binaries_if_needed() {
    log "🔨 Checking if binaries need to be built..."
    
    local any_missing=false
    local binaries=(
        "$PROVER_DIR_RELEASE/prove_sgd_stdin"
        "$PROVER_DIR_RELEASE/verify_sgd_stdin"
        "$PROVER_DIR_RELEASE/prove_lora_stdin"
        "$PROVER_DIR_RELEASE/verify_lora_stdin"
    )
    
    for binary in "${binaries[@]}"; do
        if [ ! -f "$binary" ]; then
            any_missing=true
            break
        fi
    done
    
    if [ "$any_missing" = true ]; then
        log "⚙️  Some binaries missing, attempting to build..."
        
        if [ -d "pot/zk/prover_halo2" ]; then
            cd pot/zk/prover_halo2
            
            log "🔨 Building ZK binaries in release mode..."
            if cargo build --release >> "$LOG_FILE" 2>&1; then
                log "✅ Binary build completed successfully"
                cd - > /dev/null
                return 0
            else
                log "❌ Binary build failed - check log for details"
                cd - > /dev/null
                return 1
            fi
        else
            log "❌ Rust prover directory not found at pot/zk/prover_halo2"
            return 1
        fi
    else
        log "✅ All binaries already exist"
        return 0
    fi
}

# Performance benchmark
run_performance_benchmark() {
    local prover_path="$1"
    local binary_name="$2"
    
    log "⚡ Running performance benchmark for $binary_name"
    
    local benchmark_iterations=3
    local total_time=0
    local successful_runs=0
    
    for i in $(seq 1 $benchmark_iterations); do
        log "  Benchmark iteration $i/$benchmark_iterations"
        
        # Create simple test case
        local test_input='{
            "statement": {
                "W_t_root": "'$(printf "%0*d" 64 0)'",
                "W_t1_root": "'$(printf "%0*d" 64 1)'",
                "batch_root": "'$(printf "%0*d" 64 2)'",
                "hparams_hash": "'$(printf "%0*d" 32 3)'",
                "step_nonce": 0,
                "step_number": 1,
                "epoch": 1
            },
            "witness": {
                "weights_before": [0.1, 0.2],
                "weights_after": [0.11, 0.21],
                "batch_inputs": [1.0, 2.0],
                "batch_targets": [0.5],
                "learning_rate": 0.01
            }
        }'
        
        local start_time=$(date +%s.%N)
        if echo "$test_input" | timeout 60s "$prover_path" > /dev/null 2>&1; then
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            total_time=$(echo "$total_time + $duration" | bc)
            successful_runs=$((successful_runs + 1))
            log "    Iteration $i: ${duration}s"
        else
            log "    Iteration $i: FAILED"
        fi
    done
    
    if [ $successful_runs -gt 0 ]; then
        local avg_time=$(echo "scale=3; $total_time / $successful_runs" | bc)
        log "📊 $binary_name average time: ${avg_time}s ($successful_runs/$benchmark_iterations successful)"
        echo "$avg_time" > "$TEST_DIR/${binary_name}_avg_time.txt"
    else
        log "❌ All benchmark iterations failed for $binary_name"
        return 1
    fi
}

# Generate comprehensive report
generate_report() {
    log "📋 Generating comprehensive verification report..."
    
    local report_file="$TEST_DIR/verification_report.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Collect all results
    cat > "$report_file" << EOF
{
    "timestamp": "$timestamp",
    "verification_results": {
        "rust_environment": {
            "rustc_available": $(command -v rustc >/dev/null && echo true || echo false),
            "cargo_available": $(command -v cargo >/dev/null && echo true || echo false),
            "rustc_version": "$(rustc --version 2>/dev/null || echo 'not available')",
            "cargo_version": "$(cargo --version 2>/dev/null || echo 'not available')"
        },
        "binary_status": {
            "sgd_prover": {
                "debug_exists": $([ -f "$PROVER_DIR_DEBUG/prove_sgd_stdin" ] && echo true || echo false),
                "release_exists": $([ -f "$PROVER_DIR_RELEASE/prove_sgd_stdin" ] && echo true || echo false),
                "executable": $([ -x "$PROVER_DIR_RELEASE/prove_sgd_stdin" ] && echo true || echo false)
            },
            "sgd_verifier": {
                "debug_exists": $([ -f "$PROVER_DIR_DEBUG/verify_sgd_stdin" ] && echo true || echo false),
                "release_exists": $([ -f "$PROVER_DIR_RELEASE/verify_sgd_stdin" ] && echo true || echo false),
                "executable": $([ -x "$PROVER_DIR_RELEASE/verify_sgd_stdin" ] && echo true || echo false)
            },
            "lora_prover": {
                "debug_exists": $([ -f "$PROVER_DIR_DEBUG/prove_lora_stdin" ] && echo true || echo false),
                "release_exists": $([ -f "$PROVER_DIR_RELEASE/prove_lora_stdin" ] && echo true || echo false),
                "executable": $([ -x "$PROVER_DIR_RELEASE/prove_lora_stdin" ] && echo true || echo false)
            },
            "lora_verifier": {
                "debug_exists": $([ -f "$PROVER_DIR_DEBUG/verify_lora_stdin" ] && echo true || echo false),
                "release_exists": $([ -f "$PROVER_DIR_RELEASE/verify_lora_stdin" ] && echo true || echo false),
                "executable": $([ -x "$PROVER_DIR_RELEASE/verify_lora_stdin" ] && echo true || echo false)
            }
        },
        "functional_tests": {
            "sgd_proof_generation": $([ -f "$TEST_DIR/sgd_test_proof.bin" ] && [ -s "$TEST_DIR/sgd_test_proof.bin" ] && echo true || echo false),
            "sgd_verification": $([ -f "$TEST_DIR/sgd_verify_result.txt" ] && echo true || echo false),
            "lora_proof_generation": $([ -f "$TEST_DIR/lora_test_proof.bin" ] && [ -s "$TEST_DIR/lora_test_proof.bin" ] && echo true || echo false),
            "lora_verification": $([ -f "$TEST_DIR/lora_verify_result.txt" ] && echo true || echo false)
        },
        "performance": {
            "sgd_proof_size_bytes": $([ -f "$TEST_DIR/sgd_proof_size.txt" ] && cat "$TEST_DIR/sgd_proof_size.txt" || echo null),
            "lora_proof_size_bytes": $([ -f "$TEST_DIR/lora_proof_size.txt" ] && cat "$TEST_DIR/lora_proof_size.txt" || echo null),
            "sgd_avg_time_seconds": $([ -f "$TEST_DIR/prove_sgd_stdin_avg_time.txt" ] && cat "$TEST_DIR/prove_sgd_stdin_avg_time.txt" || echo null),
            "lora_avg_time_seconds": $([ -f "$TEST_DIR/prove_lora_stdin_avg_time.txt" ] && cat "$TEST_DIR/prove_lora_stdin_avg_time.txt" || echo null)
        }
    },
    "log_file": "$LOG_FILE",
    "test_directory": "$TEST_DIR"
}
EOF
    
    log "📄 Report generated: $report_file"
    echo "📄 Full verification report: $report_file"
}

# Main verification workflow
main() {
    echo -e "${BLUE}🔍 ZK Binary Verification Suite${NC}"
    echo "=================================="
    echo ""
    
    local exit_code=0
    
    # Check Rust environment
    if ! check_rust_environment; then
        echo -e "${RED}❌ Rust environment check failed${NC}"
        exit_code=1
    fi
    
    # Build binaries if needed
    if ! build_binaries_if_needed; then
        echo -e "${YELLOW}⚠️  Binary build check had issues${NC}"
    fi
    
    echo -e "\n${BLUE}🔍 Checking Binary Availability${NC}"
    echo "================================"
    
    # Check all binaries
    local binaries=(
        "$PROVER_DIR_RELEASE/prove_sgd_stdin:SGD Prover (Release)"
        "$PROVER_DIR_RELEASE/verify_sgd_stdin:SGD Verifier (Release)"
        "$PROVER_DIR_RELEASE/prove_lora_stdin:LoRA Prover (Release)"
        "$PROVER_DIR_RELEASE/verify_lora_stdin:LoRA Verifier (Release)"
        "$PROVER_DIR_DEBUG/prove_sgd_stdin:SGD Prover (Debug)"
        "$PROVER_DIR_DEBUG/verify_sgd_stdin:SGD Verifier (Debug)"
        "$PROVER_DIR_DEBUG/prove_lora_stdin:LoRA Prover (Debug)"
        "$PROVER_DIR_DEBUG/verify_lora_stdin:LoRA Verifier (Debug)"
    )
    
    for binary_info in "${binaries[@]}"; do
        IFS=':' read -r binary_path binary_name <<< "$binary_info"
        check_binary "$binary_path" "$binary_name"
        get_binary_version "$binary_path" "$binary_name"
    done
    
    # Functional testing (only if release binaries exist)
    if [ -x "$PROVER_DIR_RELEASE/prove_sgd_stdin" ]; then
        echo -e "\n${BLUE}🧪 Functional Testing${NC}"
        echo "===================="
        
        # Test SGD workflow
        if test_sgd_prover "$PROVER_DIR_RELEASE/prove_sgd_stdin"; then
            if [ -x "$PROVER_DIR_RELEASE/verify_sgd_stdin" ]; then
                test_sgd_verifier "$PROVER_DIR_RELEASE/verify_sgd_stdin"
            fi
        else
            echo -e "${RED}❌ SGD prover test failed${NC}"
            exit_code=1
        fi
        
        # Test LoRA workflow
        if [ -x "$PROVER_DIR_RELEASE/prove_lora_stdin" ]; then
            if test_lora_prover "$PROVER_DIR_RELEASE/prove_lora_stdin"; then
                if [ -x "$PROVER_DIR_RELEASE/verify_lora_stdin" ]; then
                    test_lora_verifier "$PROVER_DIR_RELEASE/verify_lora_stdin"
                fi
            else
                echo -e "${YELLOW}⚠️  LoRA prover test failed${NC}"
            fi
        fi
        
        # Performance benchmarks (if functional tests passed)
        echo -e "\n${BLUE}⚡ Performance Benchmarks${NC}"
        echo "========================"
        
        if [ -f "$TEST_DIR/sgd_test_proof.bin" ] && [ -s "$TEST_DIR/sgd_test_proof.bin" ]; then
            run_performance_benchmark "$PROVER_DIR_RELEASE/prove_sgd_stdin" "prove_sgd_stdin"
        fi
        
        if [ -f "$TEST_DIR/lora_test_proof.bin" ] && [ -s "$TEST_DIR/lora_test_proof.bin" ]; then
            run_performance_benchmark "$PROVER_DIR_RELEASE/prove_lora_stdin" "prove_lora_stdin"
        fi
    else
        echo -e "${YELLOW}⚠️  Skipping functional tests - release binaries not available${NC}"
    fi
    
    # Generate final report
    generate_report
    
    echo ""
    echo -e "${BLUE}📊 Verification Summary${NC}"
    echo "======================="
    
    # Summary output
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ ZK binary verification completed successfully${NC}"
        echo -e "📄 Detailed log: ${LOG_FILE}"
        echo -e "📁 Test files: ${TEST_DIR}/"
    else
        echo -e "${RED}❌ ZK binary verification completed with errors${NC}"
        echo -e "📄 Check log for details: ${LOG_FILE}"
        echo -e "📁 Test files: ${TEST_DIR}/"
    fi
    
    exit $exit_code
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "ZK Binary Verification Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --build        Force rebuild of binaries"
        echo "  --no-build     Skip binary building"
        echo "  --quick        Run only basic checks"
        echo "  --report-only  Generate report from existing test data"
        echo ""
        exit 0
        ;;
    --build)
        log "🔨 Forcing rebuild of binaries..."
        rm -rf "$PROVER_DIR_RELEASE"/* "$PROVER_DIR_DEBUG"/*
        ;;
    --no-build)
        build_binaries_if_needed() { return 0; }
        ;;
    --quick)
        run_performance_benchmark() { return 0; }
        test_sgd_verifier() { return 0; }
        test_lora_verifier() { return 0; }
        ;;
    --report-only)
        generate_report
        exit 0
        ;;
esac

# Run main verification
main