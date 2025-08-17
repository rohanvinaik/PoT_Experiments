# Proof-of-Training Validation Report

## Executive Summary

This report validates the claims made in the Proof-of-Training paper through systematic testing.

**Generated**: Sun Aug 17 15:01:58 EDT 2025
**System**: Darwin arm64
**PyTorch**: 2.3.1
**Device**: mps

---

## Paper Claims Validation


### Claim 1: False Acceptance Rate < 0.1%

**Paper Reference**: Abstract, Section 3.1
**Expected Result**: Type I error rate < 0.001

**Status**: ✅ **VALIDATED**

**Test Output**:
```
✓ test_h0_stream: Type I=0.000, mean stop=69.8±11.4
✓ test_h1_stream: Type II=0.000, Power=1.00, mean stop=69.4±11.6
```

### Claim 2: False Rejection Rate < 1%

**Paper Reference**: Abstract, Section 3.1
**Expected Result**: Type II error rate < 0.01

**Status**: ✅ **VALIDATED**

**Test Output**:
```
✓ test_h1_stream: Type II=0.000, Power=1.00, mean stop=69.4±11.6
```

### Claim 3: 100% Detection of Wrapper Attacks

**Paper Reference**: Section 3.2, Table 2
**Expected Result**: All wrapper attacks detected

**Status**: ❌ **FAILED**

**Error Output**:
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: cannot import name 'detect_wrapper' from 'pot.core.wrapper_detection' (/Users/rohanvinaik/PoT_Experiments/pot/core/wrapper_detection.py)
```

### Claim 4: Sub-second Verification for Large Models

**Paper Reference**: Section 3.3
**Expected Result**: Verification time < 1000ms

**Status**: ❌ **FAILED**

**Error Output**:
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: cannot import name 'BehavioralFingerprint' from 'pot.core.fingerprint' (/Users/rohanvinaik/PoT_Experiments/pot/core/fingerprint.py)
```

### Claim 5: 2-3 Average Queries with Sequential Testing

**Paper Reference**: Section 2.4, Theorem 2.5
**Expected Result**: Mean queries between 2-3

**Status**: ❌ **FAILED**

**Error Output**:
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: cannot import name 'SequentialVerifier' from 'pot.core.sequential' (/Users/rohanvinaik/PoT_Experiments/pot/core/sequential.py)
```

### Claim 6: 99.6% Detection with 25% Challenge Leakage

**Paper Reference**: Section 3.2
**Expected Result**: Detection rate > 99% with 25% leakage

**Status**: ❌ **FAILED**

**Error Output**:
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: cannot import name 'measure_leakage' from 'pot.security.leakage' (/Users/rohanvinaik/PoT_Experiments/pot/security/leakage.py)
```

### Claim 7: Empirical-Bernstein Tighter than Hoeffding

**Paper Reference**: Section 2.4, Theorem 2.3
**Expected Result**: EB bounds tighter by 30-50%

**Status**: ❌ **FAILED**

**Error Output**:
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: cannot import name 'compare_bounds' from 'pot.core.boundaries' (/Users/rohanvinaik/PoT_Experiments/pot/core/boundaries.py)
```

### Claim 8: 90% Gas Reduction with Merkle Trees

**Paper Reference**: Section 2.2.3
**Expected Result**: Batch uses <10% gas of individual

**Status**: ⚠️ **SKIPPED** (missing dependency)
