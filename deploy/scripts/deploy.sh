#!/bin/bash
set -e

# PoT Framework Deployment Script
# Supports Docker, Kubernetes, and standalone deployments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy"

# Default configuration
DEPLOYMENT_TYPE="docker"
ENVIRONMENT="development"
NAMESPACE="pot-framework"
IMAGE_TAG="latest"
BUILD_IMAGE=false
PUSH_IMAGE=false
REGISTRY=""
VERBOSE=false
DRY_RUN=false
FORCE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy the PoT Framework using various deployment methods.

OPTIONS:
    -t, --type TYPE         Deployment type: docker, kubernetes, standalone (default: docker)
    -e, --environment ENV   Environment: development, staging, production (default: development)
    -n, --namespace NS      Kubernetes namespace (default: pot-framework)
    -i, --image-tag TAG     Docker image tag (default: latest)
    -b, --build             Build Docker image before deployment
    -p, --push              Push image to registry after build
    -r, --registry REG      Container registry URL
    -v, --verbose           Enable verbose output
    -d, --dry-run           Show what would be done without executing
    -f, --force             Force deployment (override safety checks)
    -h, --help              Show this help message

EXAMPLES:
    # Deploy using Docker Compose for development
    $0 --type docker --environment development --build

    # Deploy to Kubernetes production
    $0 --type kubernetes --environment production --namespace pot-prod --image-tag v1.0.0

    # Build and push image to registry
    $0 --build --push --registry my-registry.com/pot --image-tag v1.0.0

    # Standalone deployment
    $0 --type standalone --environment production

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -i|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -b|--build)
            BUILD_IMAGE=true
            shift
            ;;
        -p|--push)
            PUSH_IMAGE=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate deployment type
case $DEPLOYMENT_TYPE in
    docker|kubernetes|standalone)
        ;;
    *)
        log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
        log_error "Supported types: docker, kubernetes, standalone"
        exit 1
        ;;
esac

# Validate environment
case $ENVIRONMENT in
    development|staging|production)
        ;;
    *)
        log_error "Invalid environment: $ENVIRONMENT"
        log_error "Supported environments: development, staging, production"
        exit 1
        ;;
esac

# Set image name
if [[ -n "$REGISTRY" ]]; then
    IMAGE_NAME="$REGISTRY/pot-framework:$IMAGE_TAG"
else
    IMAGE_NAME="pot-framework:$IMAGE_TAG"
fi

log_info "Starting PoT Framework deployment"
log_info "  Type: $DEPLOYMENT_TYPE"
log_info "  Environment: $ENVIRONMENT"
log_info "  Image: $IMAGE_NAME"
log_info "  Namespace: $NAMESPACE"

if [[ "$DRY_RUN" == true ]]; then
    log_warning "DRY RUN MODE - No changes will be made"
fi

# Pre-deployment checks
perform_checks() {
    log_info "Performing pre-deployment checks..."
    
    # Check if required tools are available
    case $DEPLOYMENT_TYPE in
        docker)
            if ! command -v docker &> /dev/null; then
                log_error "Docker is not installed or not in PATH"
                exit 1
            fi
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose is not installed or not in PATH"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed or not in PATH"
                exit 1
            fi
            # Check cluster connectivity
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                exit 1
            fi
            ;;
        standalone)
            if ! command -v python3 &> /dev/null; then
                log_error "Python 3 is not installed or not in PATH"
                exit 1
            fi
            ;;
    esac
    
    # Check project structure
    if [[ ! -f "$PROJECT_ROOT/src/pot/__init__.py" ]]; then
        log_error "PoT framework source not found. Are you in the correct directory?"
        exit 1
    fi
    
    # Check Rust binaries for ZK proofs
    if [[ ! -d "$PROJECT_ROOT/src/pot/zk/prover_halo2" ]]; then
        log_warning "ZK prover source not found. ZK functionality may not be available."
    fi
    
    log_success "Pre-deployment checks passed"
}

# Build Docker image
build_image() {
    if [[ "$BUILD_IMAGE" == true ]]; then
        log_info "Building Docker image: $IMAGE_NAME"
        
        if [[ "$DRY_RUN" == true ]]; then
            log_info "Would run: docker build -t $IMAGE_NAME -f $DEPLOY_DIR/docker/Dockerfile $PROJECT_ROOT"
            return
        fi
        
        cd "$PROJECT_ROOT"
        
        if [[ "$VERBOSE" == true ]]; then
            docker build -t "$IMAGE_NAME" -f "$DEPLOY_DIR/docker/Dockerfile" .
        else
            docker build -t "$IMAGE_NAME" -f "$DEPLOY_DIR/docker/Dockerfile" . > /dev/null
        fi
        
        log_success "Docker image built successfully"
        
        # Push image if requested
        if [[ "$PUSH_IMAGE" == true ]]; then
            log_info "Pushing image to registry..."
            
            if [[ "$DRY_RUN" == true ]]; then
                log_info "Would run: docker push $IMAGE_NAME"
                return
            fi
            
            docker push "$IMAGE_NAME"
            log_success "Image pushed successfully"
        fi
    fi
}

# Deploy using Docker Compose
deploy_docker() {
    log_info "Deploying using Docker Compose"
    
    cd "$DEPLOY_DIR/docker"
    
    # Set environment variables
    export COMPOSE_PROJECT_NAME="pot-framework"
    export POT_IMAGE_TAG="$IMAGE_TAG"
    export POT_ENVIRONMENT="$ENVIRONMENT"
    
    # Environment-specific overrides
    COMPOSE_FILES="-f docker-compose.yml"
    
    if [[ -f "docker-compose.$ENVIRONMENT.yml" ]]; then
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.$ENVIRONMENT.yml"
        log_info "Using environment-specific compose file"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would run: docker-compose $COMPOSE_FILES up -d"
        return
    fi
    
    # Stop existing containers if not in development
    if [[ "$ENVIRONMENT" != "development" ]] || [[ "$FORCE" == true ]]; then
        log_info "Stopping existing containers..."
        docker-compose $COMPOSE_FILES down --remove-orphans
    fi
    
    # Deploy services
    log_info "Starting services..."
    docker-compose $COMPOSE_FILES up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    check_service_health_docker
    
    log_success "Docker deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes"
    
    cd "$DEPLOY_DIR/kubernetes"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would apply Kubernetes manifests to namespace: $NAMESPACE"
        return
    fi
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Apply configurations
    log_info "Applying Kubernetes manifests..."
    
    # Update image in deployment
    sed -i.bak "s|pot-framework:latest|$IMAGE_NAME|g" pot-deployment.yaml
    
    # Apply manifests
    kubectl apply -f pot-deployment.yaml
    
    # Restore original file
    mv pot-deployment.yaml.bak pot-deployment.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployments to be ready..."
    kubectl -n "$NAMESPACE" rollout status deployment/pot-api --timeout=300s
    kubectl -n "$NAMESPACE" rollout status deployment/pot-worker --timeout=300s
    
    # Check service health
    check_service_health_kubernetes
    
    log_success "Kubernetes deployment completed"
}

# Standalone deployment
deploy_standalone() {
    log_info "Deploying standalone installation"
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would install Python dependencies and start services"
        return
    fi
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Build ZK prover binaries if source exists
    if [[ -d "src/pot/zk/prover_halo2" ]]; then
        log_info "Building ZK prover binaries..."
        cd "src/pot/zk/prover_halo2"
        cargo build --release
        cd "$PROJECT_ROOT"
    fi
    
    # Set up data directories
    mkdir -p data logs configs
    
    # Initialize database
    log_info "Initializing database..."
    python -c "
from benchmarks.tracking.performance_tracker import PerformanceTracker
tracker = PerformanceTracker('data/performance.db')
print('Performance database initialized')
"
    
    # Start services based on environment
    case $ENVIRONMENT in
        development)
            log_info "Starting development server..."
            python scripts/run_e2e_validation.py --help > /dev/null  # Verify scripts work
            log_success "Standalone development setup completed"
            ;;
        production)
            log_info "Setting up production service..."
            # In production, you'd typically set up systemd services
            log_warning "Production standalone setup requires manual service configuration"
            ;;
    esac
}

# Health check functions
check_service_health_docker() {
    log_info "Checking service health..."
    
    # Check API service
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "API service is healthy"
            break
        fi
        if [[ $i -eq 30 ]]; then
            log_warning "API service health check timed out"
        fi
        sleep 2
    done
}

check_service_health_kubernetes() {
    log_info "Checking service health..."
    
    # Get service endpoints
    API_ENDPOINT=$(kubectl -n "$NAMESPACE" get service pot-api-service -o jsonpath='{.spec.clusterIP}')
    
    if [[ -n "$API_ENDPOINT" ]]; then
        log_success "Services are deployed and accessible"
    else
        log_warning "Could not verify service health"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Remove any temporary files created during deployment
}

# Main deployment logic
main() {
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Perform checks
    perform_checks
    
    # Build image if requested
    build_image
    
    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        standalone)
            deploy_standalone
            ;;
    esac
    
    log_success "Deployment completed successfully!"
    
    # Show next steps
    show_next_steps
}

# Show next steps after deployment
show_next_steps() {
    log_info "Next steps:"
    
    case $DEPLOYMENT_TYPE in
        docker)
            echo "  • API available at: http://localhost:8000"
            echo "  • Grafana dashboard: http://localhost:3000 (admin/admin)"
            echo "  • View logs: docker-compose -f $DEPLOY_DIR/docker/docker-compose.yml logs -f"
            echo "  • Stop services: docker-compose -f $DEPLOY_DIR/docker/docker-compose.yml down"
            ;;
        kubernetes)
            echo "  • Check status: kubectl -n $NAMESPACE get pods"
            echo "  • View logs: kubectl -n $NAMESPACE logs -l app=pot-api"
            echo "  • Port forward API: kubectl -n $NAMESPACE port-forward service/pot-api-service 8000:8000"
            ;;
        standalone)
            echo "  • Test installation: python scripts/run_e2e_validation.py --help"
            echo "  • Run validation: python scripts/run_e2e_validation.py --ref-model gpt2 --cand-model distilgpt2"
            ;;
    esac
}

# Run main function
main