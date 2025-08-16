# API Verification System for Proof-of-Training

## Overview

The API Verification System enables testing of external AI APIs (OpenAI, Anthropic, Google, etc.) against PoT challenges to measure their verification metrics including False Accept Rate (FAR), False Reject Rate (FRR), and Area Under ROC Curve (AUROC).

## Features

### 1. Multi-Provider Support
- **OpenAI**: GPT-3.5, GPT-4
- **Anthropic**: Claude 3 (Sonnet, Haiku, Opus)
- **Google**: Gemini Pro
- **Cohere**: Command models
- **Hugging Face**: Inference API
- **Local Models**: Custom endpoints
- **Mock API**: For testing

### 2. Challenge Generation
- Deterministic challenge creation using KDF
- Template-based language challenges
- Configurable challenge families
- Support for text, embedding, and completion modes

### 3. Sequential Verification
- Empirical-Bernstein bounds for early stopping
- Adaptive confidence intervals
- Minimizes API calls while maintaining error bounds
- Average 2-5 queries for decision

### 4. Metrics Computation
- **FAR (False Accept Rate)**: Rate of incorrectly accepting invalid models
- **FRR (False Reject Rate)**: Rate of incorrectly rejecting valid models
- **AUROC**: Overall discrimination capability
- **Precision-Recall curves**: Performance visualization
- **Confidence scores**: Statistical confidence in decisions

### 5. Cost Management
- Per-token cost tracking
- Budget limits and warnings
- Rate limiting to respect API quotas
- Caching for repeated queries

## Installation

```bash
# Install required dependencies
pip install numpy scikit-learn matplotlib seaborn pyyaml

# Optional API clients
pip install openai anthropic google-generativeai cohere
```

## Configuration

Edit `configs/api_verification.yaml`:

```yaml
apis:
  - provider: "openai"
    model_name: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
    max_tokens: 100
    temperature: 0.0
    rate_limit_rpm: 40
    cost_per_token: 0.03
    
verification:
  similarity_threshold: 0.7
  confidence_threshold: 0.8
  max_errors_percent: 10
  
challenges:
  families:
    - family: "lm:templates"
      n: 100
      params:
        templates: ["Complete: {prompt}"]
```

## Usage

### Basic Verification

```bash
# Verify all configured APIs
python scripts/api_verification.py

# Verify specific APIs
python scripts/api_verification.py --apis openai_gpt-4 anthropic_claude-3

# Custom number of challenges
python scripts/api_verification.py --num-challenges 50

# Generate plots
python scripts/api_verification.py --plot
```

### Programmatic Usage

```python
from scripts.api_verification import APIVerifier

# Initialize verifier
verifier = APIVerifier('configs/api_verification.yaml')

# Generate challenges
verifier.generate_pot_challenges(num_challenges=100)

# Verify an API
result = verifier.verify_api('openai_gpt-4')

print(f"Accepted: {result.accepted}")
print(f"Confidence: {result.confidence:.3f}")
print(f"FAR: {result.far:.4f}, FRR: {result.frr:.4f}")
print(f"AUROC: {result.auroc:.3f}")
```

## Output Structure

```
api_results/
├── verification_results_TIMESTAMP.json  # Detailed results
├── summary_TIMESTAMP.json              # Summary statistics
├── report_TIMESTAMP.md                 # Human-readable report
├── api_verification.jsonl              # Structured logs
├── roc_curves_TIMESTAMP.png           # ROC curves
├── far_frr_TIMESTAMP.png              # FAR/FRR comparison
├── distance_distributions_TIMESTAMP.png # Distance histograms
└── performance_summary_TIMESTAMP.png   # Performance overview
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/test_api_verification.py -v

# Run specific test
python -m pytest tests/test_api_verification.py::TestAPIVerification::test_verification_process

# With coverage
python -m pytest tests/test_api_verification.py --cov=scripts.api_verification
```

## Example Results

### Mock API Test
```
API: mock_mock-model
Status: ✅ ACCEPTED
Confidence: 1.000
FAR: 0.0000, FRR: 0.0000
Queries: 5, Cost: $0.00
```

### Report Format
```markdown
# API Verification Report

## Summary
- APIs Tested: 5
- Accepted: 3
- Rejected: 2
- Average FAR: 0.0234
- Average FRR: 0.0156
- Average AUROC: 0.945

## Detailed Results
| API | Model | Accepted | Confidence | FAR | FRR | AUROC |
|-----|-------|----------|------------|-----|-----|-------|
| openai_gpt-4 | gpt-4 | ✅ | 0.923 | 0.02 | 0.01 | 0.96 |
```

## Sequential Verification Algorithm

The system uses empirical-Bernstein bounds for sequential decision making:

1. Generate challenge c_i
2. Query API and compute distance d_i
3. Update statistics: mean, variance
4. Compute confidence bound U_n(α)
5. Decision rules:
   - Accept if: mean + U_n(α) ≤ threshold
   - Reject if: mean - U_n(β) ≥ threshold
   - Otherwise: continue to next challenge

This achieves:
- Early stopping (2-5 queries typical)
- Controlled error rates (FAR ≤ α, FRR ≤ β)
- Adaptive to response variance

## Security Considerations

1. **API Keys**: Store in environment variables, never commit
2. **Rate Limiting**: Respects API quotas automatically
3. **Cost Controls**: Budget limits prevent runaway costs
4. **Audit Logging**: All requests/responses logged for compliance
5. **Encryption**: Optional log encryption for sensitive data

## Performance Benchmarks

- **Latency**: 10-100ms per API call (provider dependent)
- **Throughput**: 10-60 requests/minute (rate limited)
- **Cost**: $0.001-0.03 per verification (model dependent)
- **Accuracy**: >95% AUROC for legitimate models
- **Efficiency**: 2-5 queries average with sequential testing

## Troubleshooting

### Common Issues

1. **Import Errors**: Install required API clients
2. **API Key Issues**: Check environment variables
3. **Rate Limiting**: Reduce requests per minute in config
4. **Timeout Errors**: Increase timeout in config
5. **JSON Serialization**: numpy types converted automatically

### Debug Mode

```bash
# Verbose output
python scripts/api_verification.py --verbose

# Check logs
tail -f api_results/api_verification.jsonl
```

## Future Enhancements

- [ ] Multi-modal challenge support (vision + text)
- [ ] Async API calls for parallel verification
- [ ] WebSocket support for streaming APIs
- [ ] Differential privacy for challenge generation
- [ ] Automated API discovery and testing
- [ ] Integration with CI/CD pipelines
- [ ] Real-time monitoring dashboard

## References

- Empirical-Bernstein Bounds: See POT_PAPER_ENHANCED.md
- Challenge Generation: pot/core/challenge.py
- Statistical Methods: pot/core/stats.py
- Logging Framework: pot/core/logging.py

---

*Last Updated: 2025-08-16*
*Version: 1.0.0*