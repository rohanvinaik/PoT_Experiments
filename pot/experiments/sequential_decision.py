#!/usr/bin/env python3
"""
Sequential Decision Logic for PoT Framework

Implements the Sequential Probability Ratio Test (SPRT) for efficient model verification
with early stopping criteria. Provides adaptive decision-making that minimizes the number
of challenges needed while maintaining statistical guarantees.

Key Features:
- Sequential Probability Ratio Test (SPRT) implementation
- Early stopping criteria (max queries, confidence thresholds, time limits)
- Decision tracking and comprehensive logging
- Confidence evolution monitoring
- Visualization tools for decision boundaries and ROC curves
- Paper-based default parameters for reproducible research
"""

import numpy as np
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionOutcome(Enum):
    """Possible outcomes of sequential decision process."""
    CONTINUE = "continue"      # Need more observations
    ACCEPT = "accept"          # Accept H1 (model is legitimate)
    REJECT = "reject"          # Accept H0 (model is illegitimate)
    TIMEOUT = "timeout"        # Stopped due to time limit
    MAX_QUERIES = "max_queries" # Stopped due to query limit


class StoppingReason(Enum):
    """Reasons why sequential testing stopped."""
    DECISION_REACHED = "decision_reached"       # SPRT decision boundary crossed
    MAX_QUERIES_REACHED = "max_queries_reached" # Hit query limit
    CONFIDENCE_THRESHOLD = "confidence_threshold" # Confidence threshold met
    TIME_LIMIT = "time_limit"                   # Time limit exceeded
    RESOURCE_LIMIT = "resource_limit"           # Resource constraints


@dataclass
class SPRTConfig:
    """Configuration for Sequential Probability Ratio Test."""
    alpha: float = 0.05        # Type I error rate (false positive)
    beta: float = 0.05         # Type II error rate (false negative)
    h0: float = 0.5           # Null hypothesis probability (illegitimate model)
    h1: float = 0.9           # Alternative hypothesis probability (legitimate model)
    max_queries: int = 100    # Maximum number of observations
    confidence_threshold: float = 0.95  # Stop when confidence exceeds this
    time_limit: float = 300.0  # Maximum time in seconds
    resource_limit: int = 1000  # Maximum computational resources
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.alpha < 1:
            raise ValueError(f"Alpha must be in (0,1), got {self.alpha}")
        if not 0 < self.beta < 1:
            raise ValueError(f"Beta must be in (0,1), got {self.beta}")
        if not 0 < self.h0 < self.h1 < 1:
            raise ValueError(f"Must have 0 < h0 < h1 < 1, got h0={self.h0}, h1={self.h1}")
        if self.max_queries <= 0:
            raise ValueError(f"Max queries must be positive, got {self.max_queries}")


@dataclass
class DecisionPoint:
    """Single decision point in sequential testing."""
    observation: bool          # True if challenge passed, False if failed
    log_likelihood_ratio: float # Current log likelihood ratio
    decision: DecisionOutcome  # Decision at this point
    confidence: float          # Current confidence level
    timestamp: float          # When observation was made
    cumulative_passes: int    # Total passes so far
    cumulative_fails: int     # Total fails so far
    total_observations: int   # Total observations so far
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SequentialResult:
    """Final result of sequential decision process."""
    final_decision: DecisionOutcome   # Final decision
    stopping_reason: StoppingReason  # Why testing stopped
    total_observations: int           # Total queries used
    final_confidence: float          # Final confidence level
    total_time: float               # Total time elapsed
    decision_history: List[DecisionPoint] # Complete decision history
    log_likelihood_ratio: float     # Final LLR
    evidence_for_h1: float          # Evidence supporting H1
    evidence_against_h0: float      # Evidence against H0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_decision": self.final_decision.value,
            "stopping_reason": self.stopping_reason.value,
            "total_observations": self.total_observations,
            "final_confidence": self.final_confidence,
            "total_time": self.total_time,
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "evidence_for_h1": self.evidence_for_h1,
            "evidence_against_h0": self.evidence_against_h0,
            "decision_points": len(self.decision_history),
            "metadata": self.metadata
        }


class SequentialDecisionMaker:
    """
    Sequential Probability Ratio Test (SPRT) implementation for model verification.
    
    The SPRT provides an efficient method for hypothesis testing that can make decisions
    with fewer observations than fixed-sample tests while maintaining the same error rates.
    
    Based on Wald's Sequential Probability Ratio Test with early stopping criteria
    and confidence tracking for practical deployment.
    """
    
    def __init__(self, config: Optional[SPRTConfig] = None):
        """
        Initialize Sequential Decision Maker.
        
        Args:
            config: SPRT configuration. Uses paper defaults if None.
        """
        self.config = config or SPRTConfig()
        
        # Calculate decision boundaries
        self._calculate_boundaries()
        
        # Initialize state
        self.reset()
        
        logger.info(f"Initialized SPRT with boundaries: A={self.boundary_accept:.4f}, "
                   f"B={self.boundary_reject:.4f}")
    
    def _calculate_boundaries(self):
        """Calculate SPRT decision boundaries."""
        # Standard SPRT boundaries
        self.boundary_accept = np.log((1 - self.config.beta) / self.config.alpha)
        self.boundary_reject = np.log(self.config.beta / (1 - self.config.alpha))
        
        # Log likelihood ratio for single observation
        self.llr_pass = np.log(self.config.h1 / self.config.h0)
        self.llr_fail = np.log((1 - self.config.h1) / (1 - self.config.h0))
        
        logger.debug(f"SPRT boundaries: accept={self.boundary_accept:.4f}, "
                    f"reject={self.boundary_reject:.4f}")
        logger.debug(f"LLR per observation: pass={self.llr_pass:.4f}, "
                    f"fail={self.llr_fail:.4f}")
    
    def reset(self):
        """Reset decision maker to initial state."""
        self.log_likelihood_ratio = 0.0
        self.observations = []
        self.decision_history = []
        self.cumulative_passes = 0
        self.cumulative_fails = 0
        self.start_time = time.time()
        self.is_complete = False
        self.final_result = None
        
        logger.debug("Sequential decision maker reset")
    
    def update(self, observation: bool, metadata: Optional[Dict[str, Any]] = None) -> DecisionOutcome:
        """
        Update SPRT with new observation and return decision.
        
        Args:
            observation: True if challenge passed, False if failed
            metadata: Optional metadata for this observation
            
        Returns:
            Current decision (continue, accept, reject, timeout, max_queries)
        """
        if self.is_complete:
            return self.final_result.final_decision
        
        # Record observation
        self.observations.append(observation)
        if observation:
            self.cumulative_passes += 1
            self.log_likelihood_ratio += self.llr_pass
        else:
            self.cumulative_fails += 1
            self.log_likelihood_ratio += self.llr_fail
        
        # Calculate current confidence
        confidence = self.get_confidence()
        
        # Check stopping criteria
        decision, stopping_reason = self._check_stopping_criteria()
        
        # Create decision point
        decision_point = DecisionPoint(
            observation=observation,
            log_likelihood_ratio=self.log_likelihood_ratio,
            decision=decision,
            confidence=confidence,
            timestamp=time.time(),
            cumulative_passes=self.cumulative_passes,
            cumulative_fails=self.cumulative_fails,
            total_observations=len(self.observations),
            metadata=metadata or {}
        )
        
        self.decision_history.append(decision_point)
        
        # If decision is final, create result
        if decision != DecisionOutcome.CONTINUE:
            self._finalize_result(decision, stopping_reason)
        
        logger.debug(f"Observation {len(self.observations)}: {observation}, "
                    f"LLR={self.log_likelihood_ratio:.4f}, "
                    f"confidence={confidence:.4f}, decision={decision.value}")
        
        return decision
    
    def _check_stopping_criteria(self) -> Tuple[DecisionOutcome, StoppingReason]:
        """Check all stopping criteria and return decision."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Check SPRT boundaries first
        if self.log_likelihood_ratio >= self.boundary_accept:
            return DecisionOutcome.ACCEPT, StoppingReason.DECISION_REACHED
        elif self.log_likelihood_ratio <= self.boundary_reject:
            return DecisionOutcome.REJECT, StoppingReason.DECISION_REACHED
        
        # Check early stopping criteria
        if len(self.observations) >= self.config.max_queries:
            return DecisionOutcome.MAX_QUERIES, StoppingReason.MAX_QUERIES_REACHED
        
        if elapsed_time >= self.config.time_limit:
            return DecisionOutcome.TIMEOUT, StoppingReason.TIME_LIMIT
        
        confidence = self.get_confidence()
        if confidence >= self.config.confidence_threshold:
            # Decide based on current evidence
            if self.log_likelihood_ratio > 0:
                return DecisionOutcome.ACCEPT, StoppingReason.CONFIDENCE_THRESHOLD
            else:
                return DecisionOutcome.REJECT, StoppingReason.CONFIDENCE_THRESHOLD
        
        return DecisionOutcome.CONTINUE, StoppingReason.DECISION_REACHED
    
    def get_confidence(self) -> float:
        """
        Calculate current confidence level.
        
        Returns confidence as probability that current decision is correct.
        """
        if len(self.observations) == 0:
            return 0.0
        
        # Convert log likelihood ratio to probability
        # P(H1|data) = 1 / (1 + exp(-LLR))
        if self.log_likelihood_ratio > 50:  # Prevent overflow
            return 1.0
        elif self.log_likelihood_ratio < -50:
            return 0.0
        else:
            return 1.0 / (1.0 + np.exp(-self.log_likelihood_ratio))
    
    def get_evidence_for_h1(self) -> float:
        """Get evidence supporting H1 (legitimate model)."""
        return max(0.0, self.log_likelihood_ratio)
    
    def get_evidence_against_h0(self) -> float:
        """Get evidence against H0 (illegitimate model)."""
        return max(0.0, self.log_likelihood_ratio)
    
    def _finalize_result(self, decision: DecisionOutcome, stopping_reason: StoppingReason):
        """Create final result and mark as complete."""
        self.is_complete = True
        end_time = time.time()
        
        self.final_result = SequentialResult(
            final_decision=decision,
            stopping_reason=stopping_reason,
            total_observations=len(self.observations),
            final_confidence=self.get_confidence(),
            total_time=end_time - self.start_time,
            decision_history=self.decision_history.copy(),
            log_likelihood_ratio=self.log_likelihood_ratio,
            evidence_for_h1=self.get_evidence_for_h1(),
            evidence_against_h0=self.get_evidence_against_h0(),
            metadata={
                "boundaries": {
                    "accept": self.boundary_accept,
                    "reject": self.boundary_reject
                },
                "config": {
                    "alpha": self.config.alpha,
                    "beta": self.config.beta,
                    "h0": self.config.h0,
                    "h1": self.config.h1
                }
            }
        )
        
        logger.info(f"Sequential decision complete: {decision.value} "
                   f"({stopping_reason.value}) after {len(self.observations)} observations")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of decision process."""
        return {
            "observations": len(self.observations),
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "confidence": self.get_confidence(),
            "passes": self.cumulative_passes,
            "fails": self.cumulative_fails,
            "elapsed_time": time.time() - self.start_time,
            "is_complete": self.is_complete
        }


class SequentialVisualizer:
    """Visualization tools for sequential decision process."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer with optional output directory."""
        self.output_dir = Path(output_dir) if output_dir else Path("sequential_plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def plot_decision_boundary(self, config: SPRTConfig, save_path: Optional[Path] = None) -> Path:
        """Plot SPRT decision boundaries."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate boundaries
        boundary_accept = np.log((1 - config.beta) / config.alpha)
        boundary_reject = np.log(config.beta / (1 - config.alpha))
        
        # Create observation range
        max_obs = config.max_queries
        observations = np.arange(0, max_obs + 1)
        
        # Decision regions
        ax.axhline(y=boundary_accept, color='green', linestyle='--', 
                  label=f'Accept Boundary (LLR ≥ {boundary_accept:.2f})', linewidth=2)
        ax.axhline(y=boundary_reject, color='red', linestyle='--', 
                  label=f'Reject Boundary (LLR ≤ {boundary_reject:.2f})', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, label='LLR = 0')
        
        # Fill regions
        ax.fill_between(observations, boundary_accept, max(10, boundary_accept + 2), 
                       alpha=0.3, color='green', label='Accept Region')
        ax.fill_between(observations, min(-10, boundary_reject - 2), boundary_reject, 
                       alpha=0.3, color='red', label='Reject Region')
        ax.fill_between(observations, boundary_reject, boundary_accept, 
                       alpha=0.3, color='yellow', label='Continue Region')
        
        ax.set_xlabel('Number of Observations')
        ax.set_ylabel('Log Likelihood Ratio')
        ax.set_title(f'SPRT Decision Boundaries (α={config.alpha}, β={config.beta})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "decision_boundaries.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_confidence_evolution(self, result: SequentialResult, 
                                save_path: Optional[Path] = None) -> Path:
        """Plot confidence evolution over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Extract data
        observations = range(1, len(result.decision_history) + 1)
        confidences = [dp.confidence for dp in result.decision_history]
        llrs = [dp.log_likelihood_ratio for dp in result.decision_history]
        
        # Plot confidence evolution
        ax1.plot(observations, confidences, 'b-', linewidth=2, label='Confidence')
        ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='High Confidence')
        ax1.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Neutral')
        ax1.set_xlabel('Observation Number')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Confidence Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot log likelihood ratio
        ax2.plot(observations, llrs, 'r-', linewidth=2, label='Log Likelihood Ratio')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5, label='LLR = 0')
        
        # Add boundaries if available
        if 'boundaries' in result.metadata:
            boundaries = result.metadata['boundaries']
            ax2.axhline(y=boundaries['accept'], color='green', linestyle='--', 
                       alpha=0.7, label='Accept Boundary')
            ax2.axhline(y=boundaries['reject'], color='red', linestyle='--', 
                       alpha=0.7, label='Reject Boundary')
        
        ax2.set_xlabel('Observation Number')
        ax2.set_ylabel('Log Likelihood Ratio')
        ax2.set_title('Log Likelihood Ratio Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "confidence_evolution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_roc_curves(self, results: List[SequentialResult], 
                       save_path: Optional[Path] = None) -> Path:
        """Plot ROC curves for different threshold settings."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate ROC points for different confidence thresholds
        thresholds = np.linspace(0, 1, 101)
        tpr_values = []
        fpr_values = []
        
        for threshold in thresholds:
            tp = fp = tn = fn = 0
            
            for result in results:
                # Assume ground truth is in metadata
                if 'ground_truth' in result.metadata:
                    ground_truth = result.metadata['ground_truth']
                    predicted = result.final_confidence >= threshold
                    
                    if ground_truth and predicted:
                        tp += 1
                    elif not ground_truth and predicted:
                        fp += 1
                    elif not ground_truth and not predicted:
                        tn += 1
                    else:
                        fn += 1
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Plot ROC curve
        ax.plot(fpr_values, tpr_values, 'b-', linewidth=2, label='SPRT ROC')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random Classifier')
        
        # Calculate AUC
        auc = np.trapz(tpr_values, fpr_values)
        ax.set_title(f'ROC Curve (AUC = {auc:.3f})')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "roc_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


# Paper-based default configurations
PAPER_CONFIGS = {
    "conservative": SPRTConfig(
        alpha=0.01,          # Very low false positive rate
        beta=0.01,           # Very low false negative rate
        h0=0.3,              # Conservative null hypothesis
        h1=0.9,              # High legitimate model performance
        max_queries=200,
        confidence_threshold=0.99,
        time_limit=600.0
    ),
    
    "standard": SPRTConfig(
        alpha=0.05,          # Standard significance level
        beta=0.05,           # Standard power
        h0=0.5,              # Balanced null hypothesis
        h1=0.85,             # Good legitimate model performance
        max_queries=100,
        confidence_threshold=0.95,
        time_limit=300.0
    ),
    
    "fast": SPRTConfig(
        alpha=0.1,           # Higher error rates for speed
        beta=0.1,
        h0=0.6,              # Less conservative
        h1=0.8,
        max_queries=50,
        confidence_threshold=0.9,
        time_limit=120.0
    ),
    
    "research": SPRTConfig(
        alpha=0.001,         # Research-grade precision
        beta=0.001,
        h0=0.2,              # Very conservative
        h1=0.95,             # Very high performance expectation
        max_queries=500,
        confidence_threshold=0.999,
        time_limit=1800.0
    )
}


def create_sequential_decision_maker(preset: str = "standard") -> SequentialDecisionMaker:
    """Create decision maker with preset configuration."""
    if preset not in PAPER_CONFIGS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PAPER_CONFIGS.keys())}")
    
    config = PAPER_CONFIGS[preset]
    return SequentialDecisionMaker(config)


def run_sequential_experiment(decision_maker: SequentialDecisionMaker,
                            observations: List[bool],
                            metadata_list: Optional[List[Dict[str, Any]]] = None) -> SequentialResult:
    """
    Run complete sequential experiment with given observations.
    
    Args:
        decision_maker: Initialized decision maker
        observations: List of challenge outcomes (True = pass, False = fail)
        metadata_list: Optional metadata for each observation
        
    Returns:
        Complete sequential result
    """
    decision_maker.reset()
    
    if metadata_list is None:
        metadata_list = [{}] * len(observations)
    
    final_decision = DecisionOutcome.CONTINUE
    for i, (obs, meta) in enumerate(zip(observations, metadata_list)):
        final_decision = decision_maker.update(obs, meta)
        if final_decision != DecisionOutcome.CONTINUE:
            break
    
    # If we've used all observations without a decision, force completion
    if not decision_maker.is_complete:
        # Force a decision based on current evidence
        if decision_maker.log_likelihood_ratio > 0:
            decision_maker._finalize_result(DecisionOutcome.ACCEPT, StoppingReason.MAX_QUERIES_REACHED)
        else:
            decision_maker._finalize_result(DecisionOutcome.REJECT, StoppingReason.MAX_QUERIES_REACHED)
    
    return decision_maker.final_result


def analyze_sequential_performance(results: List[SequentialResult]) -> Dict[str, Any]:
    """Analyze performance of sequential decision making."""
    if not results:
        return {}
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return {}
    
    # Basic statistics
    total_observations = [r.total_observations for r in valid_results]
    total_times = [r.total_time for r in valid_results]
    final_confidences = [r.final_confidence for r in valid_results]
    
    # Decision outcomes
    decisions = [r.final_decision for r in valid_results]
    stopping_reasons = [r.stopping_reason for r in valid_results]
    
    # Count outcomes
    decision_counts = {}
    for decision in DecisionOutcome:
        decision_counts[decision.value] = sum(1 for d in decisions if d == decision)
    
    reason_counts = {}
    for reason in StoppingReason:
        reason_counts[reason.value] = sum(1 for r in stopping_reasons if r == reason)
    
    return {
        "total_experiments": len(valid_results),
        "average_observations": np.mean(total_observations),
        "median_observations": np.median(total_observations),
        "std_observations": np.std(total_observations),
        "min_observations": np.min(total_observations),
        "max_observations": np.max(total_observations),
        "average_time": np.mean(total_times),
        "median_time": np.median(total_times),
        "average_confidence": np.mean(final_confidences),
        "median_confidence": np.median(final_confidences),
        "decision_counts": decision_counts,
        "stopping_reason_counts": reason_counts,
        "efficiency_gain": f"{(1 - np.mean(total_observations) / max(total_observations)) * 100:.1f}%"
    }


def save_sequential_results(results: List[SequentialResult], 
                          output_path: Path,
                          include_history: bool = True) -> Path:
    """Save sequential results to JSON file."""
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    output_data = {
        "summary": analyze_sequential_performance(valid_results),
        "results": []
    }
    
    for result in valid_results:
        result_dict = result.to_dict()
        if include_history:
            result_dict["decision_history"] = [
                {
                    "observation": dp.observation,
                    "log_likelihood_ratio": dp.log_likelihood_ratio,
                    "decision": dp.decision.value,
                    "confidence": dp.confidence,
                    "timestamp": dp.timestamp,
                    "total_observations": dp.total_observations
                }
                for dp in result.decision_history
            ]
        output_data["results"].append(result_dict)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    return output_path


if __name__ == "__main__":
    # Example usage
    logger.info("Testing Sequential Decision Logic")
    
    # Create decision maker with standard config
    decision_maker = create_sequential_decision_maker("standard")
    
    # Simulate some observations
    observations = [True, True, False, True, True, True, False, True]
    
    # Run experiment
    result = run_sequential_experiment(decision_maker, observations)
    
    print(f"Final decision: {result.final_decision.value}")
    print(f"Stopping reason: {result.stopping_reason.value}")
    print(f"Total observations: {result.total_observations}")
    print(f"Final confidence: {result.final_confidence:.4f}")
    print(f"Total time: {result.total_time:.4f}s")
    
    # Create visualizations
    visualizer = SequentialVisualizer()
    visualizer.plot_decision_boundary(decision_maker.config)
    visualizer.plot_confidence_evolution(result)
    
    print("Sequential decision logic test completed!")