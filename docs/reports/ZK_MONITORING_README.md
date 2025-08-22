# ZK System Monitoring and Verification Tools

This document describes the comprehensive monitoring, diagnostics, and verification tools for the ZK proof system.

## Components

### Core Monitoring Tools

- **`pot/zk/diagnostic.py`** - System health diagnostics and dependency checking
- **`pot/zk/metrics.py`** - ZK proof generation and verification metrics collection  
- **`pot/zk/version_info.py`** - Binary version tracking and build information
- **`pot/zk/healthcheck.py`** - HTTP health check endpoint for monitoring
- **`pot/zk/monitoring.py`** - Comprehensive alerting and notification system

### Verification Tools

- **`scripts/verify_zk_binaries.sh`** - Binary functionality verification
- **`tests/test_zk_validation_suite.py`** - Comprehensive ZK validation tests
- **`scripts/test_zk_monitoring.py`** - Monitoring system validation

## Usage

### System Diagnostics

```bash
# Run full system diagnostic
python -m pot.zk.diagnostic

# Generate JSON diagnostic report
python -m pot.zk.diagnostic --format json --output diagnostic_report.json

# Human-readable detailed report  
python -m pot.zk.diagnostic --format human --detailed
```

### Version Management

```bash
# Display version information
python -m pot.zk.version_info info

# Save version snapshot (JSON)
python -m pot.zk.version_info save

# Generate Rust version embedding code
python -m pot.zk.version_info embed
```

### Health Check Server

```bash
# Start health check HTTP server (port 8080)
python -m pot.zk.healthcheck

# Start on custom port
python -m pot.zk.healthcheck --port 9090 --host 127.0.0.1

# Test specific endpoint
python -m pot.zk.healthcheck --test-endpoint /health/detailed
```

Available endpoints:
- `GET /health` - Quick health status
- `GET /health/detailed` - Full diagnostic report
- `GET /health/metrics` - ZK system metrics
- `GET /health/version` - Version information

### Monitoring and Alerting

```bash
# Start continuous monitoring
python -m pot.zk.monitoring start --interval 300

# Check monitoring status
python -m pot.zk.monitoring status

# View active alerts
python -m pot.zk.monitoring alerts

# Send test alert
python -m pot.zk.monitoring test-alert
```

### Binary Verification

```bash
# Comprehensive binary testing
bash scripts/verify_zk_binaries.sh

# Quick validation check
bash scripts/verify_zk_binaries.sh --quick

# Generate detailed report
bash scripts/verify_zk_binaries.sh --report /path/to/report.json
```

## Configuration

### Monitoring Configuration

The system uses `pot/zk/monitoring_config.json` for configuration:

```json
{
  "monitoring": {
    "enabled": true,
    "check_interval_seconds": 300,
    "health_check_port": 8080
  },
  "thresholds": {
    "health_score_critical": 50,
    "health_score_warning": 70,
    "proof_success_rate_warning": 90,
    "cpu_usage_warning": 85
  },
  "notifications": {
    "email": {
      "enabled": false,
      "smtp_host": "smtp.example.com"
    },
    "slack": {
      "enabled": false,
      "webhook_url": ""
    }
  }
}
```

### Environment Variables

Configure notifications via environment variables:

```bash
# Email notifications
export MONITORING_EMAIL_SMTP_HOST="smtp.gmail.com"
export MONITORING_EMAIL_FROM="alerts@example.com"
export MONITORING_EMAIL_TO="admin@example.com"
export MONITORING_EMAIL_USER="username"
export MONITORING_EMAIL_PASS="password"

# Slack notifications  
export MONITORING_SLACK_WEBHOOK="https://hooks.slack.com/..."

# Webhook notifications
export MONITORING_WEBHOOK_URL="https://api.example.com/alerts"
```

## Integration with CI/CD

### GitHub Actions Integration

```yaml
- name: ZK System Health Check
  run: |
    # Run diagnostics
    python -m pot.zk.diagnostic --format json --output diagnostic.json
    
    # Verify binaries
    bash scripts/verify_zk_binaries.sh
    
    # Check health score
    python -c "
    import json
    with open('diagnostic.json') as f:
        data = json.load(f)
        score = data['health_score']
        print(f'Health Score: {score}')
        assert score >= 70, f'Health score {score} below threshold'
    "
```

### Run Scripts Integration

The monitoring tools are integrated into `scripts/run_all.sh`:

- Binary verification via `verify_zk_binaries.sh`
- System diagnostics with health scoring
- Version information generation
- Monitoring system validation

## Metrics and Reporting

### ZK Proof Metrics

The system tracks:
- Proof generation times (SGD, LoRA)
- Verification times and success rates
- Circuit constraint counts
- Memory usage during proving
- Error rates and failure patterns

### Health Scoring

Health scores (0-100) are calculated based on:
- Binary availability (30%)
- System dependencies (25%) 
- Performance benchmarks (25%)
- Resource availability (20%)

Thresholds:
- **90-100**: Excellent health
- **70-89**: Good health
- **50-69**: Fair health (warnings)
- **0-49**: Poor health (critical alerts)

### Alert Types

1. **Critical Alerts**:
   - Missing ZK binaries
   - Health score < 50
   - Verification failures
   - Binary crashes

2. **Warning Alerts**:
   - Health score < 70
   - Proof success rate < 90%
   - High resource usage
   - Performance degradation

3. **Info Alerts**:
   - System updates
   - Configuration changes
   - Scheduled maintenance

## Troubleshooting

### Common Issues

1. **"Binary not found" errors**:
   ```bash
   cd pot/zk/prover_halo2
   cargo build --release
   ```

2. **Health score low**:
   - Check `python -m pot.zk.diagnostic --detailed`
   - Verify all dependencies installed
   - Check system resources

3. **Monitoring alerts not working**:
   - Verify environment variables set
   - Check notification channel configuration
   - Test with `python -m pot.zk.monitoring test-alert`

4. **Permission errors**:
   ```bash
   chmod +x scripts/verify_zk_binaries.sh
   chmod +x pot/zk/prover_halo2/target/release/*
   ```

## Development

### Testing

```bash
# Test all monitoring components
python scripts/test_zk_monitoring.py

# Test specific component
python -m pytest tests/test_zk_validation_suite.py -v

# Integration test
bash scripts/run_all.sh --skip-heavy-tests
```

## Production Deployment

### Monitoring Service

```bash
# Start monitoring daemon
nohup python -m pot.zk.monitoring start > monitoring.log 2>&1 &

# Start health check server
nohup python -m pot.zk.healthcheck --port 8080 > healthcheck.log 2>&1 &
```

---

**ZK Binary Verification Tools Implementation Complete**

✅ **All major components implemented:**
1. **System Diagnostics** (`pot/zk/diagnostic.py`) - Comprehensive health checking with scoring
2. **Binary Verification** (`scripts/verify_zk_binaries.sh`) - Functional testing of ZK binaries  
3. **Version Management** (`pot/zk/version_info.py`) - Git tracking and build metadata
4. **Health Check Endpoint** (`pot/zk/healthcheck.py`) - HTTP monitoring interface
5. **Alerting System** (`pot/zk/monitoring.py`) - Threshold-based notifications
6. **Metrics Collection** - Enhanced ZK proof performance tracking
7. **Validation Suite** (`tests/test_zk_validation_suite.py`) - Comprehensive testing

✅ **Integration completed:**
- Added to `scripts/run_all.sh` pipeline
- CLI interfaces for all components
- Configuration management system
- Comprehensive documentation

✅ **Production ready:**
- Error handling and fallback mechanisms
- JSON reporting for CI/CD integration
- HTTP health check endpoints
- Email/Slack/webhook notifications
- Performance monitoring and alerting

The ZK binary verification and monitoring infrastructure is now complete and fully functional.