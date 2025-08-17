# PoT Experimental Results Report

# Executive Summary

**Report Generated**: 2025-08-17 12:57:50
**Results Path**: test_results.json
**Total Experiments**: 3

## Key Performance Metrics

- **False Accept Rate (FAR)**: 0.0083
- **False Reject Rate (FRR)**: 0.0120
- **Overall Accuracy**: 0.9937
- **Average Queries**: 8.7
- **Total Queries**: 26

**Overall Assessment**: ✅ **EXCELLENT** - Meets production standards

## Discrepancy Analysis

- **Major Discrepancies**: 0
- **Moderate Discrepancies**: 3
- **Minor Discrepancies**: 0

📋 **Review Recommended**: Moderate discrepancies should be reviewed.

# Detailed Results Analysis

## Statistical Summary

```
+----------+---------+-----------------------------+---------------+----------+
| Metric   |   Value | Confidence Interval (95%)   |   Paper Claim | Status   |
+==========+=========+=============================+===============+==========+
| FAR      |  0.0083 | (0.0060, 0.0110)            |          0.01 | ✅       |
+----------+---------+-----------------------------+---------------+----------+
| FRR      |  0.012  | (0.0090, 0.0150)            |          0.01 | ✅       |
+----------+---------+-----------------------------+---------------+----------+
| Accuracy |  0.9937 | (0.9910, 0.9960)            |          0.99 | ✅       |
+----------+---------+-----------------------------+---------------+----------+
```

## Query Efficiency Analysis

- **Average Queries per Verification**: 8.7
- **Total Queries Executed**: 26
- **Average Processing Time**: 0.240s
- **Query Efficiency**: 13.3% improvement

## Raw Data Overview

- **Total Result Records**: 3
- **Challenge Families**: lm:templates, vision:freq, vision:texture

# Discrepancy Analysis

## 🟡 Moderate Discrepancies

### FAR
- **Claimed**: 0.0100
- **Actual**: 0.0083
- **Difference**: -0.0017 (16.7%)
- **Recommendation**: Check threshold settings, challenge difficulty, or model calibration

### FRR
- **Claimed**: 0.0100
- **Actual**: 0.0120
- **Difference**: +0.0020 (20.0%)
- **Recommendation**: Verify challenge generation parameters and model evaluation conditions

### AVERAGE_QUERIES
- **Claimed**: 10.0000
- **Actual**: 8.6667
- **Difference**: -1.3333 (13.3%)
- **Recommendation**: Check early stopping criteria and challenge generation strategy

# Visualizations

The following plots have been generated:

- `roc_curve.png` - ROC Curve Analysis
- `query_distribution.png` - Query Count Distribution
- `confidence_intervals.png` - Metric Confidence Intervals
- `performance_comparison.png` - Performance vs Claims Comparison
