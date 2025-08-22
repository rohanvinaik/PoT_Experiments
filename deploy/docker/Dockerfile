# PoT Framework Docker Image
# Multi-stage build for production deployment

# Stage 1: Rust builder for ZK circuits
FROM rust:1.88 as rust-builder

WORKDIR /build

# Copy Rust ZK circuit sources
COPY src/pot/zk/prover_halo2/ ./src/pot/zk/prover_halo2/

# Build ZK prover binaries
WORKDIR /build/src/pot/zk/prover_halo2
RUN cargo build --release

# Stage 2: Python base with system dependencies
FROM python:3.11-slim as python-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 potuser && \
    mkdir -p /app /data /logs && \
    chown -R potuser:potuser /app /data /logs

# Stage 3: Production image
FROM python-base as production

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    POT_DOCKER_MODE=true \
    POT_DATA_DIR=/data \
    POT_LOGS_DIR=/logs \
    POT_CONFIG_DIR=/app/configs

# Copy ZK binaries from rust builder
COPY --from=rust-builder /build/src/pot/zk/prover_halo2/target/release/prove_sgd /usr/local/bin/
COPY --from=rust-builder /build/src/pot/zk/prover_halo2/target/release/verify_sgd /usr/local/bin/
COPY --from=rust-builder /build/src/pot/zk/prover_halo2/target/release/prove_lora /usr/local/bin/

# Make binaries executable
RUN chmod +x /usr/local/bin/prove_sgd /usr/local/bin/verify_sgd /usr/local/bin/prove_lora

# Switch to non-root user
USER potuser
WORKDIR /app

# Copy Python requirements
COPY --chown=potuser:potuser requirements.txt .
COPY --chown=potuser:potuser requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt && \
    pip install --no-cache-dir --user -r requirements-docker.txt

# Copy application code
COPY --chown=potuser:potuser src/ ./src/
COPY --chown=potuser:potuser scripts/ ./scripts/
COPY --chown=potuser:potuser configs/ ./configs/
COPY --chown=potuser:potuser tests/ ./tests/
COPY --chown=potuser:potuser *.py ./
COPY --chown=potuser:potuser *.md ./

# Create entrypoint script
COPY --chown=potuser:potuser deploy/docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Expose ports
EXPOSE 8000 8080

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]