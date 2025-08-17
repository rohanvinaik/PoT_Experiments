# PoT Experimental Results Report

# Executive Summary

**Report Generated**: 2025-08-17 12:59:42
**Results Path**: extended_results.json
**Total Experiments**: 5

## Key Performance Metrics

- **False Accept Rate (FAR)**: 0.0082
- **False Reject Rate (FRR)**: 0.0120
- **Overall Accuracy**: 0.9938
- **Average Queries**: 8.4
- **Total Queries**: 42

**Overall Assessment**: ✅ **EXCELLENT** - Meets production standards

## Discrepancy Analysis

- **Major Discrepancies**: 0
- **Moderate Discrepancies**: 2
- **Minor Discrepancies**: 1

📋 **Review Recommended**: Moderate discrepancies should be reviewed.

# Detailed Results Analysis

## Statistical Summary

```
+----------+---------+-----------------------------+---------------+----------+
| Metric   |   Value | Confidence Interval (95%)   |   Paper Claim | Status   |
+==========+=========+=============================+===============+==========+
| FAR      |  0.0082 | (0.0068, 0.0096)            |          0.01 | ✅       |
+----------+---------+-----------------------------+---------------+----------+
| FRR      |  0.012  | (0.0102, 0.0136)            |          0.01 | ✅       |
+----------+---------+-----------------------------+---------------+----------+
| Accuracy |  0.9938 | (0.9922, 0.9952)            |          0.99 | ✅       |
+----------+---------+-----------------------------+---------------+----------+
```

## Query Efficiency Analysis

- **Average Queries per Verification**: 8.4
- **Total Queries Executed**: 42
- **Average Processing Time**: 0.260s
- **Query Efficiency**: -5.0% degradation

## Raw Data Overview

- **Total Result Records**: 5
- **Challenge Families**: lm:templates, vision:freq, vision:texture

# Discrepancy Analysis

## 🟡 Moderate Discrepancies

### FAR
- **Claimed**: 0.0100
- **Actual**: 0.0082
- **Difference**: -0.0018 (18.0%)
- **Recommendation**: Check threshold settings, challenge difficulty, or model calibration

### FRR
- **Claimed**: 0.0100
- **Actual**: 0.0120
- **Difference**: +0.0020 (20.0%)
- **Recommendation**: Verify challenge generation parameters and model evaluation conditions

## 🟢 Minor Discrepancies

### AVERAGE_QUERIES
- **Claimed**: 8.0000
- **Actual**: 8.4000
- **Difference**: +0.4000 (5.0%)
- **Recommendation**: Check early stopping criteria and challenge generation strategy

# Visualizations

The following plots have been generated:

- `roc_curve.png` - ROC Curve Analysis
- `query_distribution.png` - Query Count Distribution
- `confidence_intervals.png` - Metric Confidence Intervals
- `performance_comparison.png` - Performance vs Claims Comparison
