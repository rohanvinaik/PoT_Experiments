# End-to-End Validation Pipeline

## Overview

The E2E Validation Pipeline provides a comprehensive framework for orchestrating the complete Proof-of-Training verification flow, from pre-commitment through evidence bundle generation and optional ZK proof creation.

## Key Components

### 1. PipelineOrchestrator (`e2e_pipeline.py`)

The main orchestration class that manages the full verification lifecycle:

- **Pre-commit phase**: HMAC-based challenge generation with cryptographic commitment
- **Challenge generation**: Creates deterministic challenges from seeds
- **Model loading**: Supports both local weights and API endpoints
- **Verification**: Runs statistical identity tests with CI tracking
- **Evidence generation**: Creates comprehensive audit bundles
- **ZK proof**: Optional zero-knowledge proof generation
- **Metrics tracking**: Memory, CPU, query counts, and timing at each stage

### 2. ReportGenerator (`reporting.py`)

Generates comprehensive HTML and JSON reports with:

- **Performance visualizations**: Memory usage, query times, CI evolution
- **Stage metrics**: Detailed timing and resource usage per stage
- **Decision summaries**: Clear presentation of verification results
- **Evidence bundle validation**: Cryptographic integrity checks

### 3. CLI Interface (`scripts/run_e2e_validation.py`)

Command-line interface for easy pipeline execution:

```bash
# Basic usage
python scripts/run_e2e_validation.py \
    --ref-model gpt2 \
    --cand-model distilgpt2 \
    --mode audit

# Dry-run mode for testing
python scripts/run_e2e_validation.py \
    --ref-model gpt2 \
    --cand-model distilgpt2 \
    --mode quick \
    --dry-run

# Benchmark mode
python scripts/run_e2e_validation.py \
    --ref-model model1 \
    --cand-model model2 \
    --benchmark \
    --benchmark-runs 5
```

## Testing Modes

### QUICK
- 10-120 queries
- 97.5% confidence level
- Fast initial validation

### AUDIT (Default)
- 30-400 queries
- 99% confidence level
- Production-grade verification

### EXTENDED
- 50-800 queries
- 99.9% confidence level
- Maximum confidence validation

## Verification Modes

### LOCAL_WEIGHTS
- Direct model file access
- Hash-based identity binding
- Full weight verification

### API_BLACK_BOX
- Remote endpoint verification
- No weight access required
- TEE/vendor attestation binding

### HYBRID
- Combination of local and API
- Flexible deployment options

## Output Structure

```
outputs/validation_reports/
├── evidence_bundle_*.json    # Complete evidence package
├── pipeline_results_*.json   # Detailed execution metrics
├── report_*.html             # Visual HTML report
└── summary_*.json            # Quick summary of results
```

## Pipeline Stages

1. **INITIALIZATION**: Setup and configuration
2. **PRE_COMMIT**: Challenge seed generation and commitment
3. **CHALLENGE_GENERATION**: Create actual test challenges
4. **MODEL_LOADING**: Load or connect to models
5. **VERIFICATION**: Run statistical identity tests
6. **EVIDENCE_GENERATION**: Create audit bundle
7. **REPORTING**: Generate reports and visualizations
8. **ZK_PROOF**: Optional cryptographic proof generation
9. **COMPLETED**: Pipeline finished successfully

## Integration with Existing Framework

The pipeline seamlessly integrates with:
- `src.pot.core.diff_decision.EnhancedSequentialTester`
- `src.pot.core.challenge` for challenge generation
- `src.pot.zk` for zero-knowledge proofs
- Existing TestingMode configurations

## Metrics Tracked

### Per-Stage Metrics
- Duration (seconds)
- Peak memory usage (MB)
- Memory delta
- CPU utilization
- Query count
- CI progression tracking
- Error collection

### Overall Metrics
- Total pipeline duration
- Peak memory across all stages
- Total query count
- Final confidence interval
- Decision and confidence level

## Evidence Bundle Structure

```json
{
    "run_id": "validation_20240101_120000",
    "timestamp": "2024-01-01T12:00:00",
    "config": { ... },
    "pre_commit": {
        "commitment": "...",
        "seeds": [ ... ]
    },
    "challenges": [ ... ],
    "verification": {
        "decision": "SAME/DIFFERENT",
        "confidence": 0.99,
        "n_queries": 32,
        "ci_progression": [ ... ]
    },
    "metrics": { ... },
    "environment": { ... },
    "hash": "sha256_of_bundle"
}
```

## Testing

Run integration tests:
```bash
python -m pytest tests/test_e2e_pipeline.py -v
```

Test coverage includes:
- Pipeline stage transitions
- Metric collection accuracy
- Report generation completeness
- Evidence bundle integrity
- Error handling

## Configuration

### Via CLI Arguments
```bash
python scripts/run_e2e_validation.py \
    --mode audit \
    --n-challenges 50 \
    --max-queries 500 \
    --output-dir custom/output \
    --enable-zk
```

### Via Config File
```json
{
    "mode": "audit",
    "n_challenges": 50,
    "max_queries": 500,
    "output_dir": "custom/output",
    "enable_zk": true
}
```

```bash
python scripts/run_e2e_validation.py \
    --config config.json \
    --ref-model model1 \
    --cand-model model2
```

## Performance Benchmarking

The pipeline includes built-in benchmarking capabilities:

```bash
# Run multiple iterations for performance testing
python scripts/run_e2e_validation.py \
    --ref-model gpt2 \
    --cand-model distilgpt2 \
    --benchmark \
    --benchmark-runs 10
```

Benchmark results include:
- Average duration per run
- Average peak memory usage
- Average query count
- Individual run details
- Success rate

## Visualization Examples

The HTML reports include:
- **Memory usage by stage**: Bar chart showing peak memory per pipeline stage
- **Query response times**: Line plot with rolling average
- **CI evolution**: Confidence interval narrowing over time
- **Stage duration distribution**: Pie chart of time spent in each stage

## Best Practices

1. **Always use dry-run first**: Test configuration before actual runs
2. **Monitor memory usage**: Large models may require memory tracking
3. **Save evidence bundles**: Keep for audit trail and reproducibility
4. **Use appropriate mode**: QUICK for development, AUDIT for production
5. **Enable ZK proofs**: For cryptographic verification attestation

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure running from PoT_Experiments root directory
2. **Memory issues**: Use `--disable-memory-tracking` for constrained systems
3. **Model loading failures**: Verify model paths and permissions
4. **JSON serialization errors**: Usually Path objects - fixed in latest version

### Debug Mode

Enable verbose output for debugging:
```bash
python scripts/run_e2e_validation.py \
    --ref-model model1 \
    --cand-model model2 \
    --verbose
```

## API Reference

### PipelineOrchestrator

```python
from src.pot.validation import PipelineOrchestrator, PipelineConfig

config = PipelineConfig(
    testing_mode=TestingMode.AUDIT_GRADE,
    verification_mode=VerificationMode.LOCAL_WEIGHTS,
    enable_zk_proof=True,
    output_dir=Path("custom/output")
)

orchestrator = PipelineOrchestrator(config)
results = orchestrator.run_complete_pipeline(
    ref_model_path="path/to/model1",
    cand_model_path="path/to/model2",
    n_challenges=32
)
```

### ReportGenerator

```python
from src.pot.validation import ReportGenerator

generator = ReportGenerator(output_dir=Path("reports"))
report_path = generator.generate_html_report(
    pipeline_results=results,
    evidence_bundle=bundle
)
```

## Future Enhancements

- [ ] Distributed execution for large-scale validation
- [ ] Real-time monitoring dashboard
- [ ] Integration with cloud storage for evidence bundles
- [ ] Support for multi-model comparison matrices
- [ ] Advanced statistical analysis options
- [ ] Automated model discovery and validation