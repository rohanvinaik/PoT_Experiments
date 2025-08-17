# Proof-of-Training Metrics Discrepancy Report
Generated: 2025-08-17T12:44:52.900293

## Overall Assessment
**Minor discrepancies observed**

## Detailed Metric Comparisons
+-----------------+------------+-----------+-------------+---------+---------------+
| Metric          |   Measured |   Claimed | Rel. Diff   | In CI   | Significant   |
+=================+============+===========+=============+=========+===============+
| FAR             |     0.011  |      0.01 | 10.2%       | ✓       | ✗             |
+-----------------+------------+-----------+-------------+---------+---------------+
| FRR             |     0.0076 |      0.01 | 23.9%       | ✓       | ✗             |
+-----------------+------------+-----------+-------------+---------+---------------+
| Accuracy        |     0.991  |      0.99 | 0.1%        | ✓       | ✗             |
+-----------------+------------+-----------+-------------+---------+---------------+
| Efficiency Gain |     0.9105 |      0.9  | 1.2%        | ✗       | ✗             |
+-----------------+------------+-----------+-------------+---------+---------------+
| Average Queries |     8.947  |     10    | 10.5%       | ✗       | ✗             |
+-----------------+------------+-----------+-------------+---------+---------------+

## Significant Discrepancies
- Average Queries: measured 8.9470 vs claimed 10.0000 (diff: 10.5%)

## Recommendations
- Investigate causes of discrepancies
- Consider increasing sample size for more precise estimates
