# PoT Experimental Results Report

# Executive Summary

**Report Generated**: 2025-08-17 12:59:44
**Results Path**: custom_results.json
**Total Experiments**: 2

## Key Performance Metrics

- **False Accept Rate (FAR)**: 0.0070
- **False Reject Rate (FRR)**: 0.0135
- **Overall Accuracy**: 0.9925
- **Average Queries**: 10.0
- **Total Queries**: 20

**Overall Assessment**: ✅ **EXCELLENT** - Meets production standards

## Discrepancy Analysis

- **Major Discrepancies**: 2
- **Moderate Discrepancies**: 0
- **Minor Discrepancies**: 0

⚠️ **Action Required**: Major discrepancies detected that require investigation.

# Detailed Results Analysis

## Statistical Summary

```
+----------+---------+-----------------------------+---------------+----------+
| Metric   |   Value | Confidence Interval (95%)   |   Paper Claim | Status   |
+==========+=========+=============================+===============+==========+
| FAR      |  0.007  | (0.0060, 0.0080)            |          0.01 | ✅       |
+----------+---------+-----------------------------+---------------+----------+
| FRR      |  0.0135 | (0.0120, 0.0150)            |          0.01 | ✅       |
+----------+---------+-----------------------------+---------------+----------+
| Accuracy |  0.9925 | (0.9910, 0.9940)            |          0.99 | ✅       |
+----------+---------+-----------------------------+---------------+----------+
```

## Query Efficiency Analysis

- **Average Queries per Verification**: 10.0
- **Total Queries Executed**: 20
- **Average Processing Time**: 0.261s
- **Query Efficiency**: 0.0% degradation

## Raw Data Overview

- **Total Result Records**: 2
- **Challenge Families**: vision:freq, vision:texture

# Discrepancy Analysis

## 🔴 Major Discrepancies

### FAR
- **Claimed**: 0.0100
- **Actual**: 0.0070
- **Difference**: -0.0030 (30.0%)
- **Recommendation**: Check threshold settings, challenge difficulty, or model calibration

### FRR
- **Claimed**: 0.0100
- **Actual**: 0.0135
- **Difference**: +0.0035 (35.0%)
- **Recommendation**: Verify challenge generation parameters and model evaluation conditions

# Visualizations

The following plots have been generated:

- `roc_curve.png` - ROC Curve Analysis
- `query_distribution.png` - Query Count Distribution
- `confidence_intervals.png` - Metric Confidence Intervals
- `performance_comparison.png` - Performance vs Claims Comparison
