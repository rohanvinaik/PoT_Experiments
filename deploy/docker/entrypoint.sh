#!/bin/bash
set -e

# PoT Framework Docker Entrypoint Script

# Environment setup
export PATH="/home/potuser/.local/bin:$PATH"

# Create necessary directories
mkdir -p $POT_DATA_DIR $POT_LOGS_DIR
mkdir -p $POT_DATA_DIR/models $POT_DATA_DIR/results $POT_DATA_DIR/cache
mkdir -p $POT_LOGS_DIR/api $POT_LOGS_DIR/verification $POT_LOGS_DIR/audit

echo "🚀 Starting PoT Framework Container"
echo "   Mode: $1"
echo "   Data Dir: $POT_DATA_DIR"
echo "   Logs Dir: $POT_LOGS_DIR"
echo "   Config Dir: $POT_CONFIG_DIR"

# Verify ZK binaries are available
echo "🔧 Checking ZK prover binaries..."
if command -v prove_sgd >/dev/null 2>&1; then
    echo "   ✅ prove_sgd available"
else
    echo "   ❌ prove_sgd not found"
fi

if command -v verify_sgd >/dev/null 2>&1; then
    echo "   ✅ verify_sgd available" 
else
    echo "   ❌ verify_sgd not found"
fi

# Initialize database if needed
if [ ! -f "$POT_DATA_DIR/pot.db" ]; then
    echo "🗄️  Initializing database..."
    python -c "
from src.pot.audit.validation.audit_validator import AuditValidator
from benchmarks.tracking.performance_tracker import PerformanceTracker

# Initialize audit database
validator = AuditValidator()
print('✅ Audit database initialized')

# Initialize performance tracking database
tracker = PerformanceTracker('$POT_DATA_DIR/performance.db')
print('✅ Performance database initialized')
"
fi

# Function to run API server
run_api() {
    echo "🌐 Starting API server..."
    exec python scripts/api/start_server.py \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info \
        --data-dir $POT_DATA_DIR \
        --config-dir $POT_CONFIG_DIR
}

# Function to run verification worker
run_worker() {
    echo "⚙️  Starting verification worker..."
    exec python scripts/workers/verification_worker.py \
        --data-dir $POT_DATA_DIR \
        --log-level info \
        --worker-id ${HOSTNAME:-worker}
}

# Function to run E2E validation
run_validation() {
    echo "🔍 Running E2E validation..."
    
    # Default parameters
    REF_MODEL=${REF_MODEL:-"gpt2"}
    CAND_MODEL=${CAND_MODEL:-"distilgpt2"}
    MODE=${MODE:-"audit"}
    OUTPUT_DIR=${OUTPUT_DIR:-"$POT_DATA_DIR/validation_results"}
    
    exec python scripts/run_e2e_validation.py \
        --ref-model "$REF_MODEL" \
        --cand-model "$CAND_MODEL" \
        --mode "$MODE" \
        --output-dir "$OUTPUT_DIR" \
        --verbose
}

# Function to run batch processing
run_batch() {
    echo "📦 Running batch processing..."
    
    MANIFEST_FILE=${MANIFEST_FILE:-"manifests/batch_validation.yaml"}
    OUTPUT_DIR=${OUTPUT_DIR:-"$POT_DATA_DIR/batch_results"}
    
    if [ ! -f "$MANIFEST_FILE" ]; then
        echo "❌ Manifest file not found: $MANIFEST_FILE"
        echo "   Available manifests:"
        ls -la manifests/ || echo "   No manifests directory found"
        exit 1
    fi
    
    exec bash scripts/run_all.sh "$MANIFEST_FILE" "$OUTPUT_DIR"
}

# Function to run benchmarks
run_benchmarks() {
    echo "📊 Running benchmarks..."
    
    BENCHMARK_TYPE=${BENCHMARK_TYPE:-"performance"}
    ITERATIONS=${ITERATIONS:-5}
    OUTPUT_DIR=${OUTPUT_DIR:-"$POT_DATA_DIR/benchmark_results"}
    
    case $BENCHMARK_TYPE in
        "performance")
            exec python scripts/benchmark_performance.py \
                --iterations $ITERATIONS \
                --output-dir "$OUTPUT_DIR"
            ;;
        "zk")
            exec python scripts/run_zk_benchmarks.py performance \
                --iterations $ITERATIONS \
                --output-dir "$OUTPUT_DIR"
            ;;
        "security")
            exec python scripts/benchmark_security.py \
                --output-dir "$OUTPUT_DIR"
            ;;
        *)
            echo "❌ Unknown benchmark type: $BENCHMARK_TYPE"
            echo "   Available types: performance, zk, security"
            exit 1
            ;;
    esac
}

# Function to run health check
run_health_check() {
    echo "🏥 Running health check..."
    exec python scripts/health_check.py --full
}

# Function to run shell/debugging mode
run_shell() {
    echo "🐚 Starting shell for debugging..."
    exec /bin/bash
}

# Main command dispatch
case "$1" in
    "api")
        run_api
        ;;
    "worker")
        run_worker
        ;;
    "validation")
        run_validation
        ;;
    "batch")
        run_batch
        ;;
    "benchmarks")
        run_benchmarks
        ;;
    "health")
        run_health_check
        ;;
    "shell"|"bash")
        run_shell
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        echo "Available commands:"
        echo "  api         - Start REST API server"
        echo "  worker      - Start verification worker"
        echo "  validation  - Run E2E validation"
        echo "  batch       - Run batch processing"
        echo "  benchmarks  - Run benchmarks"
        echo "  health      - Run health check"
        echo "  shell       - Start interactive shell"
        echo ""
        echo "Environment variables:"
        echo "  REF_MODEL     - Reference model for validation (default: gpt2)"
        echo "  CAND_MODEL    - Candidate model for validation (default: distilgpt2)"
        echo "  MODE          - Validation mode: quick|audit|extended (default: audit)"
        echo "  MANIFEST_FILE - Batch manifest file (default: manifests/batch_validation.yaml)"
        echo "  OUTPUT_DIR    - Output directory (default: \$POT_DATA_DIR/results)"
        echo "  BENCHMARK_TYPE- Benchmark type: performance|zk|security (default: performance)"
        echo "  ITERATIONS    - Number of benchmark iterations (default: 5)"
        echo ""
        exit 1
        ;;
esac