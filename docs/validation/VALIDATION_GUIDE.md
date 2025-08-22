# PoT Framework Validation Guide

## How Tests Validate Paper Claims

This document maps each test in our validation suite to specific claims made in the Proof-of-Training paper.

## Quick Validation Command

```bash
# Run validation with detailed output
bash scripts/run_validation_report.sh

# View results
cat test_results/validation_report_latest.md
```

## Paper Claims and Their Validation

### Claim 1: False Acceptance Rate < 0.1%
**Paper Section**: Abstract, Section 3.1  
**Validation Tests**:
- `pot.core.test_sequential_verify` - Tests Type I error rates
- `pot.core.test_fingerprint` - Validates fingerprint uniqueness
- `scripts/run_verify_enhanced.py` - End-to-end FAR measurement

**Expected Results**: FAR should be < 0.001 across all test scenarios

---

### Claim 2: False Rejection Rate < 1%
**Paper Section**: Abstract, Section 3.1  
**Validation Tests**:
- `pot.core.test_sequential_verify` - Tests Type II error rates
- `pot.security.test_proof_of_training` - Validates legitimate model acceptance
- E1 experiment in `experimental_report.py`

**Expected Results**: FRR should be < 0.01 for identical models

---

### Claim 3: 100% Detection of Wrapper Attacks
**Paper Section**: Section 3.2, Table 2  
**Validation Tests**:
- `pot.core.test_wrapper_detection` - Wrapper attack detection
- `scripts/run_attack_simulator.py --attack wrapper` - Simulated wrapper attacks
- E2 experiment with wrapper scenarios

**Expected Results**: All wrapper attacks detected (100% detection rate)

---

### Claim 4: Sub-second Verification for 7B+ Parameters
**Paper Section**: Section 3.3, Performance Benchmarks  
**Validation Tests**:
- `pot.lm.test_time_tolerance` - LLM verification timing
- E3 experiment with large models
- Performance benchmarks in `scripts/run_all_comprehensive.sh`

**Expected Results**: Verification time < 1000ms for 7B parameter models

---

### Claim 5: 2-3 Average Queries with Sequential Testing
**Paper Section**: Section 2.4, Theorem 2.5  
**Validation Tests**:
- `pot.core.test_sequential_verify` - Sequential stopping analysis
- E4 experiment measuring query counts
- `test_anytime_validity` in sequential tests

**Expected Results**: Mean queries = 2-3 for clear accept/reject cases

---

### Claim 6: 99.6% Detection with 25% Challenge Leakage
**Paper Section**: Section 3.2, Attack Resistance  
**Validation Tests**:
- `pot.security.test_leakage` - Leakage resistance testing
- E2 experiment with ρ=0.25 leakage
- Attack suite with partial information

**Expected Results**: Detection rate > 99% even with 25% leaked challenges

---

### Claim 7: Empirical-Bernstein Bounds Tighter than Hoeffding
**Paper Section**: Section 2.4, Theorem 2.3  
**Validation Tests**:
- `pot.core.test_boundaries` - Bound comparison tests
- Statistical analysis in `experimental_report.py`
- Confidence interval measurements

**Expected Results**: EB bounds 30-50% tighter when variance is small

---

### Claim 8: Blockchain Integration with 90% Gas Reduction
**Paper Section**: Section 2.2.3, Merkle Tree Verification  
**Validation Tests**:
- `pot.audit.test_merkle` - Merkle tree batch verification
- `pot.security.test_blockchain_client` - Gas usage measurements
- Batch vs individual transaction comparison

**Expected Results**: Batch verification uses <10% gas of individual transactions

---

## Test Suite Organization

### Core Module Tests
Located in `pot/core/`:
- `test_sequential_verify.py` - Statistical verification
- `test_fingerprint.py` - Behavioral fingerprinting
- `test_prf.py` - Pseudorandom function security
- `test_wrapper_detection.py` - Attack detection
- `test_boundaries.py` - Statistical bounds

### Security Tests
Located in `pot/security/`:
- `test_proof_of_training.py` - Main protocol
- `test_leakage.py` - Information leakage
- `test_integrated.py` - End-to-end security

### Experimental Validation (E1-E7)
Run via `scripts/experimental_report.py`:
- **E1**: Coverage-Separation (FAR/FRR validation)
- **E2**: Attack Resistance (all attack types)
- **E3**: Large-Scale Models (performance at scale)
- **E4**: Sequential Testing (query efficiency)
- **E5**: API Verification (black-box validation)
- **E6**: Regulatory Compliance (baseline comparison)
- **E7**: Component Ablation (probe effectiveness)

## Running Validation Tests

### Quick Validation (~30 seconds)
```bash
bash scripts/run_all_quick.sh
```
**Validates**: Basic functionality, core imports, configuration files

### Standard Validation (~5 minutes)
```bash
bash scripts/run_all.sh
```
**Validates**: All core claims, statistical properties, basic attacks

### Comprehensive Validation (~30 minutes)
```bash
bash scripts/run_all_comprehensive.sh
```
**Validates**: All paper claims, extensive attack scenarios, performance benchmarks

### Generate Detailed Report
```bash
python scripts/generate_validation_report.py --output test_results/
```
**Produces**: 
- `validation_report.md` - Human-readable results
- `validation_metrics.json` - Machine-readable metrics
- `claim_validation.csv` - Claim-by-claim status

## Understanding Test Output

### Success Indicators
- ✅ **GREEN checkmarks**: Test passed, claim validated
- 📊 **Metrics match**: Measured values within claimed bounds
- ⏱️ **Performance met**: Timing requirements satisfied

### Warning Signs
- ⚠️ **YELLOW warnings**: Non-critical issues (e.g., no GPU)
- 🔄 **Timeout**: Test still running (not necessarily failed)
- ℹ️ **Info messages**: Configuration notes, not errors

### Failure Indicators
- ❌ **RED X marks**: Test failed, investigate
- 📉 **Metrics exceeded**: Values outside claimed bounds
- 🚫 **Import errors**: Missing dependencies

## Interpreting Results

### Example Success Output
```
Testing Sequential Verification... ✓
  Type I error: 0.0008 < 0.001 ✓ (Claim 1 validated)
  Type II error: 0.0042 < 0.01 ✓ (Claim 2 validated)
  Mean queries: 2.7 ✓ (Claim 5 validated)
```

### Example Warning Output
```
Testing Large Model Performance... ⚠️
  Verification time: 1.2s (CPU-only mode)
  Note: Claim 4 requires GPU for sub-second performance
```

## Validation Report Structure

The automated validation report includes:

1. **Executive Summary**
   - Overall validation rate (e.g., 21/22 = 95.5%)
   - Critical claims status
   - Test environment details

2. **Claim-by-Claim Validation**
   - Paper reference
   - Test methodology
   - Measured results
   - Pass/Fail status

3. **Detailed Test Results**
   - Individual test outputs
   - Performance metrics
   - Error analysis

4. **Reproducibility Information**
   - System configuration
   - Random seeds used
   - Dependency versions

## Troubleshooting Validation Failures

### Common Issues and Solutions

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| Import errors | Missing dependencies | Run `pip install -r requirements.txt` |
| Timeout on E3 | No GPU available | Expected on CPU; not a failure |
| High FAR/FRR | Wrong threshold | Check tau values in configs |
| Merkle test fails | Missing web3 | Install with `pip install web3` |
| Attack detection < 100% | Partial run | Use `--comprehensive` flag |

### Getting Help

1. Check `test_results/validation_report_*.log` for details
2. Review relevant test file for expected behavior
3. Consult paper section for theoretical background
4. Open issue with validation report attached

## Continuous Validation

For development:
```bash
# Run after each change
pytest pot/core/test_sequential_verify.py -v

# Run before commit
bash scripts/run_all_quick.sh

# Run before release
bash scripts/run_all_comprehensive.sh
```

## Summary

Each test in our suite directly validates specific paper claims. The validation system provides:
- **Traceability**: Every claim linked to specific tests
- **Transparency**: Clear success/failure criteria
- **Reproducibility**: Deterministic validation process
- **Completeness**: All major claims covered

Run `bash scripts/run_validation_report.sh` for a complete validation report with all claims checked.