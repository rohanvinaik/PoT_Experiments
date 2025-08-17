#!/usr/bin/env python3
"""
Metrics Calculator for PoT Framework

Comprehensive metrics calculation, statistical testing, and validation for
Proof-of-Training experiments. Provides rigorous statistical analysis including
False Accept Rate (FAR), False Reject Rate (FRR), confidence intervals,
significance testing, and comparison with paper claims.

Key Features:
- FAR/FRR calculation with multiple aggregation methods
- Bootstrap confidence intervals and statistical significance testing
- Cross-validation metrics and performance analysis
- Paper claims comparison and discrepancy reporting
- Result validation and consistency checks
- Formatted tables matching academic paper style
- Suspicious pattern detection and flagging
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import json
import logging
from pathlib import Path

# Optional imports for enhanced functionality
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Try to import scipy.bootstrap if available (newer scipy versions)
try:
    from scipy import bootstrap
    SCIPY_BOOTSTRAP_AVAILABLE = True
except ImportError:
    SCIPY_BOOTSTRAP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be calculated."""
    FAR = "false_accept_rate"          # False Accept Rate
    FRR = "false_reject_rate"          # False Reject Rate  
    ACCURACY = "accuracy"              # Overall accuracy
    PRECISION = "precision"            # Positive predictive value
    RECALL = "recall"                  # True positive rate
    F1_SCORE = "f1_score"             # Harmonic mean of precision/recall
    SPECIFICITY = "specificity"        # True negative rate
    NPV = "negative_predictive_value"  # Negative predictive value
    AUC = "area_under_curve"          # Area under ROC curve
    AVERAGE_QUERIES = "average_queries" # Mean number of queries
    EFFICIENCY_GAIN = "efficiency_gain" # Improvement over fixed testing


class AggregationMethod(Enum):
    """Methods for aggregating metrics across experiments."""
    MEAN = "mean"                      # Arithmetic mean
    MEDIAN = "median"                  # Median value
    HARMONIC_MEAN = "harmonic_mean"    # Harmonic mean
    WEIGHTED_MEAN = "weighted_mean"    # Weighted by sample size
    BOOTSTRAP_MEAN = "bootstrap_mean"  # Bootstrap estimate


class StatisticalTest(Enum):
    """Statistical tests for comparing metrics."""
    T_TEST = "t_test"                  # Student's t-test
    WILCOXON = "wilcoxon"             # Wilcoxon signed-rank test
    MANN_WHITNEY = "mann_whitney"      # Mann-Whitney U test
    BOOTSTRAP_TEST = "bootstrap_test"  # Bootstrap hypothesis test
    PERMUTATION_TEST = "permutation_test" # Permutation test


@dataclass
class MetricResult:
    """Single metric calculation result."""
    metric_type: MetricType
    value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    standard_error: float
    method: AggregationMethod
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        ci_lower, ci_upper = self.confidence_interval
        return (f"{self.metric_type.value}: {self.value:.4f} "
                f"[{ci_lower:.4f}, {ci_upper:.4f}] (n={self.sample_size})")


@dataclass
class StatisticalTestResult:
    """Result of statistical significance test."""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    power: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        significance = "significant" if self.is_significant else "not significant"
        return (f"{self.test_type.value}: statistic={self.statistic:.4f}, "
                f"p={self.p_value:.4f} ({significance})")


@dataclass
class PaperClaims:
    """Paper claims for comparison."""
    far: float = 0.01              # Claimed False Accept Rate
    frr: float = 0.01              # Claimed False Reject Rate
    accuracy: float = 0.99         # Claimed accuracy
    efficiency_gain: float = 0.90  # Claimed efficiency improvement
    average_queries: float = 10    # Claimed average queries
    confidence_level: float = 0.95 # Claimed confidence level
    
    # Additional context
    dataset: str = "unknown"
    model_type: str = "unknown"
    experimental_setup: str = "standard"
    paper_reference: str = "PoT Paper"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "far": self.far,
            "frr": self.frr,
            "accuracy": self.accuracy,
            "efficiency_gain": self.efficiency_gain,
            "average_queries": self.average_queries,
            "confidence_level": self.confidence_level,
            "dataset": self.dataset,
            "model_type": self.model_type,
            "experimental_setup": self.experimental_setup,
            "paper_reference": self.paper_reference
        }


@dataclass
class DiscrepancyReport:
    """Report of discrepancies between claims and measurements."""
    metric_comparisons: Dict[str, Dict[str, Any]]
    significant_discrepancies: List[str]
    suspicious_patterns: List[str]
    overall_assessment: str
    recommendations: List[str]
    timestamp: str
    
    def __str__(self) -> str:
        lines = [f"Discrepancy Report ({self.timestamp})", "=" * 50]
        lines.append(f"Overall Assessment: {self.overall_assessment}")
        
        if self.significant_discrepancies:
            lines.append(f"\nSignificant Discrepancies ({len(self.significant_discrepancies)}):")
            for disc in self.significant_discrepancies:
                lines.append(f"  • {disc}")
        
        if self.suspicious_patterns:
            lines.append(f"\nSuspicious Patterns ({len(self.suspicious_patterns)}):")
            for pattern in self.suspicious_patterns:
                lines.append(f"  • {pattern}")
        
        if self.recommendations:
            lines.append(f"\nRecommendations ({len(self.recommendations)}):")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        
        return "\n".join(lines)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for PoT framework experiments.
    
    Provides rigorous statistical analysis including FAR/FRR calculation,
    confidence intervals, significance testing, and comparison with paper claims.
    """
    
    def __init__(self, confidence_level: float = 0.95, 
                 bootstrap_samples: int = 10000,
                 random_seed: Optional[int] = None):
        """
        Initialize metrics calculator.
        
        Args:
            confidence_level: Confidence level for intervals (default 0.95)
            bootstrap_samples: Number of bootstrap samples (default 10000)
            random_seed: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Cache for expensive calculations
        self._cache = {}
        
        logger.info(f"Initialized MetricsCalculator with confidence_level={confidence_level}, "
                   f"bootstrap_samples={bootstrap_samples}")
    
    def calculate_far(self, predictions: np.ndarray, labels: np.ndarray,
                     method: AggregationMethod = AggregationMethod.MEAN) -> MetricResult:
        """
        Calculate False Accept Rate (FAR).
        
        FAR = False Positives / (False Positives + True Negatives)
        = P(predict legitimate | actually illegitimate)
        
        Args:
            predictions: Binary predictions (1 = legitimate, 0 = illegitimate)
            labels: Ground truth labels (1 = legitimate, 0 = illegitimate)
            method: Aggregation method for multiple experiments
            
        Returns:
            MetricResult with FAR calculation
        """
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have same length")
        
        # Calculate confusion matrix elements
        true_negatives = np.sum((predictions == 0) & (labels == 0))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        
        # FAR calculation
        denominator = false_positives + true_negatives
        if denominator == 0:
            logger.warning("No negative samples found, FAR undefined")
            far_value = 0.0
        else:
            far_value = false_positives / denominator
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_metric(
            predictions, labels, self._far_function
        )
        
        # Standard error calculation
        if denominator > 0:
            se = np.sqrt(far_value * (1 - far_value) / denominator)
        else:
            se = 0.0
        
        return MetricResult(
            metric_type=MetricType.FAR,
            value=far_value,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(predictions),
            standard_error=se,
            method=method,
            metadata={
                "true_negatives": int(true_negatives),
                "false_positives": int(false_positives),
                "total_negatives": int(denominator)
            }
        )
    
    def calculate_frr(self, predictions: np.ndarray, labels: np.ndarray,
                     method: AggregationMethod = AggregationMethod.MEAN) -> MetricResult:
        """
        Calculate False Reject Rate (FRR).
        
        FRR = False Negatives / (False Negatives + True Positives)
        = P(predict illegitimate | actually legitimate)
        
        Args:
            predictions: Binary predictions (1 = legitimate, 0 = illegitimate)
            labels: Ground truth labels (1 = legitimate, 0 = illegitimate)
            method: Aggregation method for multiple experiments
            
        Returns:
            MetricResult with FRR calculation
        """
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have same length")
        
        # Calculate confusion matrix elements
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        
        # FRR calculation
        denominator = false_negatives + true_positives
        if denominator == 0:
            logger.warning("No positive samples found, FRR undefined")
            frr_value = 0.0
        else:
            frr_value = false_negatives / denominator
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_metric(
            predictions, labels, self._frr_function
        )
        
        # Standard error calculation
        if denominator > 0:
            se = np.sqrt(frr_value * (1 - frr_value) / denominator)
        else:
            se = 0.0
        
        return MetricResult(
            metric_type=MetricType.FRR,
            value=frr_value,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(predictions),
            standard_error=se,
            method=method,
            metadata={
                "true_positives": int(true_positives),
                "false_negatives": int(false_negatives),
                "total_positives": int(denominator)
            }
        )
    
    def calculate_accuracy(self, predictions: np.ndarray, labels: np.ndarray,
                          method: AggregationMethod = AggregationMethod.MEAN) -> MetricResult:
        """Calculate overall accuracy."""
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        
        accuracy_value = np.mean(predictions == labels)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_metric(
            predictions, labels, self._accuracy_function
        )
        
        # Standard error
        se = np.sqrt(accuracy_value * (1 - accuracy_value) / len(predictions))
        
        return MetricResult(
            metric_type=MetricType.ACCURACY,
            value=accuracy_value,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(predictions),
            standard_error=se,
            method=method,
            metadata={
                "correct_predictions": int(np.sum(predictions == labels)),
                "total_predictions": len(predictions)
            }
        )
    
    def calculate_confidence_intervals(self, data: np.ndarray, 
                                     confidence: float = None,
                                     method: str = "bootstrap") -> Tuple[float, float]:
        """
        Calculate confidence intervals for data.
        
        Args:
            data: Input data array
            confidence: Confidence level (uses class default if None)
            method: Method ("bootstrap", "normal", "t", "percentile")
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if confidence is None:
            confidence = self.confidence_level
        
        data = np.asarray(data)
        
        if method == "bootstrap":
            return self._bootstrap_ci(data, confidence)
        elif method == "normal":
            return self._normal_ci(data, confidence)
        elif method == "t":
            return self._t_ci(data, confidence)
        elif method == "percentile":
            return self._percentile_ci(data, confidence)
        else:
            raise ValueError(f"Unknown CI method: {method}")
    
    def calculate_average_queries(self, stopping_times: List[int],
                                method: AggregationMethod = AggregationMethod.MEAN) -> MetricResult:
        """
        Calculate average number of queries with confidence intervals.
        
        Args:
            stopping_times: List of stopping times (number of queries)
            method: Aggregation method
            
        Returns:
            MetricResult with average queries calculation
        """
        stopping_times = np.asarray(stopping_times)
        
        if method == AggregationMethod.MEAN:
            avg_value = np.mean(stopping_times)
        elif method == AggregationMethod.MEDIAN:
            avg_value = np.median(stopping_times)
        elif method == AggregationMethod.HARMONIC_MEAN:
            avg_value = stats.hmean(stopping_times[stopping_times > 0])
        else:
            avg_value = np.mean(stopping_times)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(stopping_times, self.confidence_level)
        
        # Standard error
        se = np.std(stopping_times) / np.sqrt(len(stopping_times))
        
        return MetricResult(
            metric_type=MetricType.AVERAGE_QUERIES,
            value=avg_value,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(stopping_times),
            standard_error=se,
            method=method,
            metadata={
                "min_queries": int(np.min(stopping_times)),
                "max_queries": int(np.max(stopping_times)),
                "median_queries": float(np.median(stopping_times)),
                "std_queries": float(np.std(stopping_times))
            }
        )
    
    def calculate_efficiency_gain(self, adaptive_queries: List[int],
                                fixed_sample_size: int,
                                method: AggregationMethod = AggregationMethod.MEAN) -> MetricResult:
        """
        Calculate efficiency gain over fixed-sample testing.
        
        Efficiency gain = 1 - (mean_adaptive_queries / fixed_sample_size)
        
        Args:
            adaptive_queries: List of adaptive query counts
            fixed_sample_size: Fixed sample size for comparison
            method: Aggregation method
            
        Returns:
            MetricResult with efficiency gain calculation
        """
        adaptive_queries = np.asarray(adaptive_queries)
        
        if method == AggregationMethod.MEAN:
            mean_adaptive = np.mean(adaptive_queries)
        elif method == AggregationMethod.MEDIAN:
            mean_adaptive = np.median(adaptive_queries)
        else:
            mean_adaptive = np.mean(adaptive_queries)
        
        efficiency_value = 1.0 - (mean_adaptive / fixed_sample_size)
        
        # Bootstrap efficiency calculation
        def efficiency_func(queries):
            return 1.0 - (np.mean(queries) / fixed_sample_size)
        
        bootstrap_efficiencies = []
        for _ in range(self.bootstrap_samples):
            sample_queries = np.random.choice(adaptive_queries, 
                                           size=len(adaptive_queries), 
                                           replace=True)
            bootstrap_efficiencies.append(efficiency_func(sample_queries))
        
        ci_lower = np.percentile(bootstrap_efficiencies, 
                               (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_efficiencies,
                               (1 + self.confidence_level) / 2 * 100)
        
        # Standard error
        se = np.std(bootstrap_efficiencies)
        
        return MetricResult(
            metric_type=MetricType.EFFICIENCY_GAIN,
            value=efficiency_value,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(adaptive_queries),
            standard_error=se,
            method=method,
            metadata={
                "mean_adaptive_queries": float(mean_adaptive),
                "fixed_sample_size": fixed_sample_size,
                "percent_reduction": float(efficiency_value * 100)
            }
        )
    
    def perform_statistical_test(self, sample1: np.ndarray, sample2: np.ndarray,
                               test_type: StatisticalTest,
                               alternative: str = "two-sided") -> StatisticalTestResult:
        """
        Perform statistical significance test.
        
        Args:
            sample1: First sample
            sample2: Second sample (or null hypothesis value)
            test_type: Type of statistical test
            alternative: Alternative hypothesis ("two-sided", "less", "greater")
            
        Returns:
            StatisticalTestResult with test results
        """
        sample1 = np.asarray(sample1)
        sample2 = np.asarray(sample2)
        
        if test_type == StatisticalTest.T_TEST:
            if sample2.size == 1:
                # One-sample t-test
                statistic, p_value = stats.ttest_1samp(sample1, sample2[0], 
                                                     alternative=alternative)
            else:
                # Two-sample t-test
                statistic, p_value = stats.ttest_ind(sample1, sample2,
                                                   alternative=alternative)
        
        elif test_type == StatisticalTest.WILCOXON:
            if sample2.size == 1:
                # One-sample Wilcoxon
                statistic, p_value = stats.wilcoxon(sample1 - sample2[0],
                                                  alternative=alternative)
            else:
                # Wilcoxon rank-sum (Mann-Whitney U)
                statistic, p_value = stats.ranksums(sample1, sample2)
        
        elif test_type == StatisticalTest.MANN_WHITNEY:
            statistic, p_value = stats.mannwhitneyu(sample1, sample2,
                                                   alternative=alternative)
        
        elif test_type == StatisticalTest.BOOTSTRAP_TEST:
            statistic, p_value = self._bootstrap_hypothesis_test(
                sample1, sample2, alternative
            )
        
        elif test_type == StatisticalTest.PERMUTATION_TEST:
            statistic, p_value = self._permutation_test(
                sample1, sample2, alternative
            )
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Determine significance
        alpha = 1 - self.confidence_level
        is_significant = p_value < alpha
        
        # Calculate effect size (Cohen's d for appropriate tests)
        effect_size = None
        if test_type in [StatisticalTest.T_TEST]:
            if sample2.size == 1:
                effect_size = (np.mean(sample1) - sample2[0]) / np.std(sample1)
            else:
                pooled_std = np.sqrt(((len(sample1) - 1) * np.var(sample1) + 
                                    (len(sample2) - 1) * np.var(sample2)) / 
                                   (len(sample1) + len(sample2) - 2))
                effect_size = (np.mean(sample1) - np.mean(sample2)) / pooled_std
        
        return StatisticalTestResult(
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            metadata={
                "alternative": alternative,
                "sample1_size": len(sample1),
                "sample2_size": len(sample2),
                "alpha": alpha
            }
        )
    
    def compare_to_paper_claims(self, measured_metrics: Dict[str, MetricResult],
                              paper_claims: PaperClaims) -> DiscrepancyReport:
        """
        Compare measured metrics to paper claims.
        
        Args:
            measured_metrics: Dictionary of measured MetricResults
            paper_claims: Paper claims to compare against
            
        Returns:
            DiscrepancyReport with detailed comparison
        """
        from datetime import datetime
        
        comparisons = {}
        significant_discrepancies = []
        suspicious_patterns = []
        
        # Map measured metrics to paper claims
        claim_mapping = {
            "far": ("FAR", paper_claims.far),
            "frr": ("FRR", paper_claims.frr),
            "accuracy": ("Accuracy", paper_claims.accuracy),
            "efficiency_gain": ("Efficiency Gain", paper_claims.efficiency_gain),
            "average_queries": ("Average Queries", paper_claims.average_queries)
        }
        
        for claim_key, (metric_name, claimed_value) in claim_mapping.items():
            # Find corresponding measured metric
            measured_result = None
            for key, result in measured_metrics.items():
                if claim_key.lower() in key.lower() or metric_name.lower() in key.lower():
                    measured_result = result
                    break
            
            if measured_result is None:
                suspicious_patterns.append(f"Missing measurement for {metric_name}")
                continue
            
            # Perform statistical test
            measured_values = np.array([measured_result.value])
            claimed_values = np.array([claimed_value])
            
            test_result = self.perform_statistical_test(
                measured_values, claimed_values, StatisticalTest.T_TEST
            )
            
            # Calculate relative difference
            relative_diff = abs(measured_result.value - claimed_value) / claimed_value
            
            # Check if claim is within confidence interval
            ci_lower, ci_upper = measured_result.confidence_interval
            claim_in_ci = ci_lower <= claimed_value <= ci_upper
            
            comparison = {
                "measured_value": measured_result.value,
                "claimed_value": claimed_value,
                "relative_difference": relative_diff,
                "absolute_difference": abs(measured_result.value - claimed_value),
                "confidence_interval": measured_result.confidence_interval,
                "claim_in_ci": claim_in_ci,
                "statistical_test": test_result,
                "sample_size": measured_result.sample_size
            }
            
            comparisons[metric_name] = comparison
            
            # Flag significant discrepancies
            if not claim_in_ci and relative_diff > 0.1:  # >10% difference
                significant_discrepancies.append(
                    f"{metric_name}: measured {measured_result.value:.4f} vs "
                    f"claimed {claimed_value:.4f} (diff: {relative_diff:.1%})"
                )
            
            # Flag suspicious patterns
            if relative_diff > 0.5:  # >50% difference
                suspicious_patterns.append(
                    f"{metric_name} shows extreme discrepancy (>{relative_diff:.1%})"
                )
        
        # Additional suspicious pattern detection
        far_result = measured_metrics.get("far")
        frr_result = measured_metrics.get("frr")
        
        if far_result and frr_result:
            if far_result.value == 0.0 and frr_result.value == 0.0:
                suspicious_patterns.append("Both FAR and FRR are exactly zero (unrealistic)")
            
            if far_result.value > 0.5 or frr_result.value > 0.5:
                suspicious_patterns.append("Error rates exceed 50% (poor performance)")
        
        # Overall assessment
        if len(significant_discrepancies) == 0:
            overall_assessment = "Results consistent with paper claims"
        elif len(significant_discrepancies) <= 2:
            overall_assessment = "Minor discrepancies observed"
        else:
            overall_assessment = "Major discrepancies detected"
        
        # Generate recommendations
        recommendations = []
        if significant_discrepancies:
            recommendations.append("Investigate causes of discrepancies")
            recommendations.append("Consider increasing sample size for more precise estimates")
        
        if suspicious_patterns:
            recommendations.append("Review experimental setup and data quality")
            recommendations.append("Verify implementation correctness")
        
        if not recommendations:
            recommendations.append("Results appear valid and consistent")
        
        return DiscrepancyReport(
            metric_comparisons=comparisons,
            significant_discrepancies=significant_discrepancies,
            suspicious_patterns=suspicious_patterns,
            overall_assessment=overall_assessment,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def generate_discrepancy_report(self, discrepancy_report: DiscrepancyReport,
                                  output_path: Optional[Path] = None) -> str:
        """
        Generate formatted discrepancy report.
        
        Args:
            discrepancy_report: DiscrepancyReport object
            output_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("# Proof-of-Training Metrics Discrepancy Report")
        lines.append(f"Generated: {discrepancy_report.timestamp}")
        lines.append("")
        
        # Overall assessment
        lines.append("## Overall Assessment")
        lines.append(f"**{discrepancy_report.overall_assessment}**")
        lines.append("")
        
        # Detailed comparisons table
        if discrepancy_report.metric_comparisons:
            lines.append("## Detailed Metric Comparisons")
            
            table_data = []
            headers = ["Metric", "Measured", "Claimed", "Rel. Diff", "In CI", "Significant"]
            
            for metric_name, comparison in discrepancy_report.metric_comparisons.items():
                measured = comparison["measured_value"]
                claimed = comparison["claimed_value"]
                rel_diff = comparison["relative_difference"]
                in_ci = "✓" if comparison["claim_in_ci"] else "✗"
                significant = "✓" if comparison["statistical_test"].is_significant else "✗"
                
                table_data.append([
                    metric_name,
                    f"{measured:.4f}",
                    f"{claimed:.4f}",
                    f"{rel_diff:.1%}",
                    in_ci,
                    significant
                ])
            
            if TABULATE_AVAILABLE:
                lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
            else:
                # Fallback table formatting
                lines.append(" | ".join(headers))
                lines.append("-" * (len(" | ".join(headers))))
                for row in table_data:
                    lines.append(" | ".join(str(cell) for cell in row))
            lines.append("")
        
        # Significant discrepancies
        if discrepancy_report.significant_discrepancies:
            lines.append("## Significant Discrepancies")
            for disc in discrepancy_report.significant_discrepancies:
                lines.append(f"- {disc}")
            lines.append("")
        
        # Suspicious patterns
        if discrepancy_report.suspicious_patterns:
            lines.append("## Suspicious Patterns")
            for pattern in discrepancy_report.suspicious_patterns:
                lines.append(f"- {pattern}")
            lines.append("")
        
        # Recommendations
        if discrepancy_report.recommendations:
            lines.append("## Recommendations")
            for rec in discrepancy_report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        report_text = "\n".join(lines)
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Discrepancy report saved to {output_path}")
        
        return report_text
    
    def validate_results(self, metrics: Dict[str, MetricResult]) -> List[str]:
        """
        Validate metric results for impossible values and inconsistencies.
        
        Args:
            metrics: Dictionary of metric results
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check for impossible values
        for name, result in metrics.items():
            value = result.value
            metric_type = result.metric_type
            
            # Probability metrics should be in [0, 1]
            if metric_type in [MetricType.FAR, MetricType.FRR, MetricType.ACCURACY,
                             MetricType.PRECISION, MetricType.RECALL, MetricType.SPECIFICITY]:
                if not (0 <= value <= 1):
                    warnings.append(f"{name}: value {value:.4f} outside [0,1] range")
            
            # Confidence intervals should contain the estimate
            ci_lower, ci_upper = result.confidence_interval
            if not (ci_lower <= value <= ci_upper):
                warnings.append(f"{name}: estimate {value:.4f} outside CI [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Check for suspiciously perfect values
            if value in [0.0, 1.0] and result.sample_size > 10:
                warnings.append(f"{name}: suspiciously perfect value {value}")
            
            # Check for extremely narrow confidence intervals
            ci_width = ci_upper - ci_lower
            if ci_width < 0.001 and result.sample_size < 1000:
                warnings.append(f"{name}: unexpectedly narrow CI (width={ci_width:.6f})")
        
        # Cross-metric consistency checks
        far_result = metrics.get("far")
        frr_result = metrics.get("frr")
        accuracy_result = metrics.get("accuracy")
        
        if far_result and frr_result and accuracy_result:
            # Check if accuracy is consistent with error rates
            # This is an approximation assuming balanced classes
            expected_accuracy = 1.0 - 0.5 * (far_result.value + frr_result.value)
            accuracy_diff = abs(accuracy_result.value - expected_accuracy)
            
            if accuracy_diff > 0.1:  # 10% difference
                warnings.append(f"Accuracy inconsistent with error rates "
                              f"(accuracy={accuracy_result.value:.3f}, "
                              f"expected≈{expected_accuracy:.3f})")
        
        return warnings
    
    def format_results_table(self, metrics: Dict[str, MetricResult],
                           paper_claims: Optional[PaperClaims] = None,
                           style: str = "academic") -> str:
        """
        Format results in academic paper style table.
        
        Args:
            metrics: Dictionary of metric results
            paper_claims: Optional paper claims for comparison
            style: Table style ("academic", "simple", "markdown")
            
        Returns:
            Formatted table string
        """
        if style == "academic":
            return self._format_academic_table(metrics, paper_claims)
        elif style == "simple":
            return self._format_simple_table(metrics, paper_claims)
        elif style == "markdown":
            return self._format_markdown_table(metrics, paper_claims)
        else:
            raise ValueError(f"Unknown table style: {style}")
    
    def _format_academic_table(self, metrics: Dict[str, MetricResult],
                             paper_claims: Optional[PaperClaims] = None) -> str:
        """Format table in academic paper style."""
        lines = []
        lines.append("Table: Experimental Results Summary")
        lines.append("=" * 50)
        
        # Prepare table data
        table_data = []
        headers = ["Metric", "Value", "95% CI", "n", "SE"]
        
        if paper_claims:
            headers.extend(["Claimed", "Diff"])
        
        # Sort metrics for consistent ordering
        metric_order = ["far", "frr", "accuracy", "average_queries", "efficiency_gain"]
        sorted_metrics = []
        
        for order_key in metric_order:
            for name, result in metrics.items():
                if order_key in name.lower():
                    sorted_metrics.append((name, result))
                    break
        
        # Add remaining metrics
        for name, result in metrics.items():
            if not any(name == sorted_name for sorted_name, _ in sorted_metrics):
                sorted_metrics.append((name, result))
        
        for name, result in sorted_metrics:
            ci_lower, ci_upper = result.confidence_interval
            row = [
                name.upper(),
                f"{result.value:.4f}",
                f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                str(result.sample_size),
                f"{result.standard_error:.4f}"
            ]
            
            if paper_claims:
                # Add claimed value and difference if available
                claimed_value = getattr(paper_claims, name.lower(), None)
                if claimed_value is not None:
                    diff = result.value - claimed_value
                    row.extend([f"{claimed_value:.4f}", f"{diff:+.4f}"])
                else:
                    row.extend(["—", "—"])
            
            table_data.append(row)
        
        # Format table
        if TABULATE_AVAILABLE:
            table_str = tabulate(table_data, headers=headers, tablefmt="grid", 
                               floatfmt=".4f", numalign="center", stralign="center")
            lines.append(table_str)
        else:
            # Fallback table formatting
            lines.append(" | ".join(headers))
            lines.append("-" * (len(" | ".join(headers))))
            for row in table_data:
                lines.append(" | ".join(str(cell) for cell in row))
        lines.append("")
        lines.append("Note: CI = Confidence Interval, SE = Standard Error")
        
        return "\n".join(lines)
    
    def _format_simple_table(self, metrics: Dict[str, MetricResult],
                           paper_claims: Optional[PaperClaims] = None) -> str:
        """Format simple table."""
        table_data = []
        headers = ["Metric", "Value", "CI Lower", "CI Upper"]
        
        for name, result in metrics.items():
            ci_lower, ci_upper = result.confidence_interval
            table_data.append([
                name,
                f"{result.value:.4f}",
                f"{ci_lower:.4f}",
                f"{ci_upper:.4f}"
            ])
        
        if TABULATE_AVAILABLE:
            return tabulate(table_data, headers=headers, tablefmt="simple")
        else:
            # Fallback simple table
            lines = [" ".join(f"{h:<12}" for h in headers)]
            lines.append("-" * 50)
            for row in table_data:
                lines.append(" ".join(f"{str(cell):<12}" for cell in row))
            return "\n".join(lines)
    
    def _format_markdown_table(self, metrics: Dict[str, MetricResult],
                             paper_claims: Optional[PaperClaims] = None) -> str:
        """Format markdown table."""
        table_data = []
        headers = ["Metric", "Value", "95% CI", "n"]
        
        for name, result in metrics.items():
            ci_lower, ci_upper = result.confidence_interval
            table_data.append([
                name,
                f"{result.value:.4f}",
                f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                str(result.sample_size)
            ])
        
        if TABULATE_AVAILABLE:
            return tabulate(table_data, headers=headers, tablefmt="pipe")
        else:
            # Fallback markdown table
            lines = ["| " + " | ".join(headers) + " |"]
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in table_data:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            return "\n".join(lines)
    
    # Helper methods for specific metric calculations
    def _far_function(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Helper function for FAR calculation."""
        true_negatives = np.sum((predictions == 0) & (labels == 0))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        denominator = false_positives + true_negatives
        return false_positives / denominator if denominator > 0 else 0.0
    
    def _frr_function(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Helper function for FRR calculation."""
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        denominator = false_negatives + true_positives
        return false_negatives / denominator if denominator > 0 else 0.0
    
    def _accuracy_function(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Helper function for accuracy calculation."""
        return np.mean(predictions == labels)
    
    def _bootstrap_metric(self, predictions: np.ndarray, labels: np.ndarray,
                         metric_func) -> Tuple[float, float]:
        """Bootstrap confidence interval for metric."""
        bootstrap_values = []
        
        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
            sample_pred = predictions[indices]
            sample_labels = labels[indices]
            
            # Calculate metric for this sample
            metric_value = metric_func(sample_pred, sample_labels)
            bootstrap_values.append(metric_value)
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_values, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)
        
        return ci_lower, ci_upper
    
    def _bootstrap_ci(self, data: np.ndarray, confidence: float) -> Tuple[float, float]:
        """Bootstrap confidence interval for data mean."""
        bootstrap_means = []
        
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        return ci_lower, ci_upper
    
    def _normal_ci(self, data: np.ndarray, confidence: float) -> Tuple[float, float]:
        """Normal approximation confidence interval."""
        mean = np.mean(data)
        se = np.std(data) / np.sqrt(len(data))
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        margin = z_score * se
        return mean - margin, mean + margin
    
    def _t_ci(self, data: np.ndarray, confidence: float) -> Tuple[float, float]:
        """T-distribution confidence interval."""
        mean = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(len(data))
        df = len(data) - 1
        t_score = stats.t.ppf((1 + confidence) / 2, df)
        
        margin = t_score * se
        return mean - margin, mean + margin
    
    def _percentile_ci(self, data: np.ndarray, confidence: float) -> Tuple[float, float]:
        """Percentile-based confidence interval."""
        alpha = 1 - confidence
        ci_lower = np.percentile(data, alpha / 2 * 100)
        ci_upper = np.percentile(data, (1 - alpha / 2) * 100)
        
        return ci_lower, ci_upper
    
    def _bootstrap_hypothesis_test(self, sample1: np.ndarray, sample2: np.ndarray,
                                 alternative: str) -> Tuple[float, float]:
        """Bootstrap hypothesis test."""
        observed_diff = np.mean(sample1) - np.mean(sample2)
        
        # Pool samples under null hypothesis
        pooled = np.concatenate([sample1, sample2])
        n1, n2 = len(sample1), len(sample2)
        
        bootstrap_diffs = []
        for _ in range(self.bootstrap_samples):
            # Resample from pooled distribution
            resampled = np.random.choice(pooled, size=len(pooled), replace=True)
            resample1 = resampled[:n1]
            resample2 = resampled[n1:]
            
            bootstrap_diffs.append(np.mean(resample1) - np.mean(resample2))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value based on alternative hypothesis
        if alternative == "two-sided":
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        elif alternative == "greater":
            p_value = np.mean(bootstrap_diffs >= observed_diff)
        elif alternative == "less":
            p_value = np.mean(bootstrap_diffs <= observed_diff)
        
        return observed_diff, p_value
    
    def _permutation_test(self, sample1: np.ndarray, sample2: np.ndarray,
                        alternative: str) -> Tuple[float, float]:
        """Permutation test."""
        observed_diff = np.mean(sample1) - np.mean(sample2)
        
        # Combine samples
        combined = np.concatenate([sample1, sample2])
        n1 = len(sample1)
        
        permutation_diffs = []
        for _ in range(self.bootstrap_samples):
            # Randomly permute combined samples
            permuted = np.random.permutation(combined)
            perm_sample1 = permuted[:n1]
            perm_sample2 = permuted[n1:]
            
            permutation_diffs.append(np.mean(perm_sample1) - np.mean(perm_sample2))
        
        permutation_diffs = np.array(permutation_diffs)
        
        # Calculate p-value
        if alternative == "two-sided":
            p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        elif alternative == "greater":
            p_value = np.mean(permutation_diffs >= observed_diff)
        elif alternative == "less":
            p_value = np.mean(permutation_diffs <= observed_diff)
        
        return observed_diff, p_value


# Convenience functions for common PoT paper claims
POT_PAPER_CLAIMS = PaperClaims(
    far=0.01,              # 1% false accept rate
    frr=0.01,              # 1% false reject rate  
    accuracy=0.99,         # 99% accuracy
    efficiency_gain=0.90,  # 90% efficiency gain
    average_queries=10,    # 10 average queries
    confidence_level=0.95, # 95% confidence
    dataset="CIFAR-10/BERT",
    model_type="Vision/Language",
    experimental_setup="standard",
    paper_reference="Proof-of-Training Paper"
)


def create_metrics_calculator(confidence_level: float = 0.95,
                            bootstrap_samples: int = 10000,
                            random_seed: Optional[int] = None) -> MetricsCalculator:
    """Create metrics calculator with specified parameters."""
    return MetricsCalculator(
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed
    )


def calculate_all_metrics(predictions: np.ndarray, labels: np.ndarray,
                        stopping_times: Optional[List[int]] = None,
                        fixed_sample_size: Optional[int] = None,
                        calculator: Optional[MetricsCalculator] = None) -> Dict[str, MetricResult]:
    """
    Calculate all standard PoT metrics.
    
    Args:
        predictions: Binary predictions array
        labels: Ground truth labels array
        stopping_times: List of stopping times (optional)
        fixed_sample_size: Fixed sample size for efficiency calculation (optional)
        calculator: MetricsCalculator instance (creates new if None)
        
    Returns:
        Dictionary of calculated metrics
    """
    if calculator is None:
        calculator = create_metrics_calculator()
    
    metrics = {}
    
    # Basic classification metrics
    metrics["far"] = calculator.calculate_far(predictions, labels)
    metrics["frr"] = calculator.calculate_frr(predictions, labels)
    metrics["accuracy"] = calculator.calculate_accuracy(predictions, labels)
    
    # Query efficiency metrics
    if stopping_times is not None:
        metrics["average_queries"] = calculator.calculate_average_queries(stopping_times)
        
        if fixed_sample_size is not None:
            metrics["efficiency_gain"] = calculator.calculate_efficiency_gain(
                stopping_times, fixed_sample_size
            )
    
    return metrics


if __name__ == "__main__":
    # Example usage
    logger.info("Testing Metrics Calculator")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate predictions and labels
    labels = np.random.binomial(1, 0.6, n_samples)  # 60% legitimate
    predictions = labels.copy()
    
    # Add some errors
    error_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    predictions[error_indices] = 1 - predictions[error_indices]
    
    # Create stopping times
    stopping_times = np.random.poisson(8, n_samples).tolist()
    
    # Calculate metrics
    calculator = create_metrics_calculator()
    metrics = calculate_all_metrics(predictions, labels, stopping_times, 20)
    
    # Print results
    print("Calculated Metrics:")
    for name, result in metrics.items():
        print(f"  {result}")
    
    # Compare to paper claims
    discrepancy_report = calculator.compare_to_paper_claims(metrics, POT_PAPER_CLAIMS)
    print(f"\n{discrepancy_report}")
    
    # Validate results
    validation_warnings = calculator.validate_results(metrics)
    if validation_warnings:
        print(f"\nValidation Warnings:")
        for warning in validation_warnings:
            print(f"  - {warning}")
    
    # Format table
    table = calculator.format_results_table(metrics, POT_PAPER_CLAIMS)
    print(f"\n{table}")
    
    print("Metrics calculator test completed!")