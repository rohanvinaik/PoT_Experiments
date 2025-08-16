#!/usr/bin/env python3
"""
Script to compare PoT against baseline methods
"""

import argparse
import json
import numpy as np
import time
from pathlib import Path
import sys
import os
from typing import Dict, Any, List, Tuple
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.core.jsonenc import safe_json_dump
from pot.core.utils import sha256_bytes
from pot.eval.baselines import (
    benign_input_fingerprint,
    adversarial_trajectory_fingerprint,
    fixed_n_aggregation_distance,
    sequential_hoeffding_test,
)


class BaselineMethod:
    """Base class for baseline verification methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.metadata = {}
    
    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        """Verify if outputs are from same model"""
        raise NotImplementedError


class NaiveHashBaseline(BaselineMethod):
    """Exact hash matching baseline"""
    
    def __init__(self):
        super().__init__("naive_hash")
    
    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        """Exact hash comparison"""
        start_time = time.time()
        
        # Convert outputs to bytes and hash
        bytes1 = str(outputs1).encode('utf-8')
        bytes2 = str(outputs2).encode('utf-8')
        
        hash1 = sha256_bytes(bytes1)
        hash2 = sha256_bytes(bytes2)
        
        exact_match = hash1 == hash2
        
        return {
            "method": self.name,
            "decision": "accept" if exact_match else "reject",
            "confidence": 1.0 if exact_match else 0.0,
            "time": time.time() - start_time,
            "metadata": {
                "hash1": hash1[:16] + "...",
                "hash2": hash2[:16] + "...",
                "exact_match": exact_match
            }
        }


class SimpleDistanceBaseline(BaselineMethod):
    """Simple distance thresholding baseline"""
    
    def __init__(self, threshold: float = 0.1, metric: str = "l2"):
        super().__init__(f"simple_distance_{metric}")
        self.threshold = threshold
        self.metric = metric
    
    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        """Distance-based verification"""
        start_time = time.time()
        
        # Convert to numpy arrays
        arr1 = np.array(outputs1, dtype=np.float32)
        arr2 = np.array(outputs2, dtype=np.float32)
        
        # Flatten if needed
        if arr1.ndim > 1:
            arr1 = arr1.flatten()
        if arr2.ndim > 1:
            arr2 = arr2.flatten()
        
        # Ensure same size
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
        
        # Compute distance
        if self.metric == "l2":
            distance = np.linalg.norm(arr1 - arr2)
        elif self.metric == "l1":
            distance = np.abs(arr1 - arr2).sum()
        elif self.metric == "cosine":
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            if norm1 > 0 and norm2 > 0:
                distance = 1 - np.dot(arr1, arr2) / (norm1 * norm2)
            else:
                distance = 1.0
        else:
            distance = np.linalg.norm(arr1 - arr2)
        
        # Normalize distance to [0, 1] confidence
        confidence = np.exp(-distance / self.threshold)
        accept = distance < self.threshold
        
        return {
            "method": self.name,
            "decision": "accept" if accept else "reject",
            "confidence": float(confidence),
            "time": time.time() - start_time,
            "metadata": {
                "distance": float(distance),
                "threshold": self.threshold,
                "metric": self.metric
            }
        }


class StatisticalBaseline(BaselineMethod):
    """Statistical similarity baseline"""
    
    def __init__(self, alpha: float = 0.05):
        super().__init__("statistical")
        self.alpha = alpha
    
    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        """Statistical hypothesis testing"""
        start_time = time.time()
        
        from scipy import stats
        
        # Convert to numpy arrays
        arr1 = np.array(outputs1, dtype=np.float32).flatten()
        arr2 = np.array(outputs2, dtype=np.float32).flatten()
        
        # Ensure same size
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(arr1, arr2)
        
        # T-test (if applicable)
        if len(arr1) > 1 and len(arr2) > 1:
            t_stat, t_pvalue = stats.ttest_ind(arr1, arr2)
        else:
            t_stat, t_pvalue = 0, 1
        
        # Decision based on p-values
        accept = ks_pvalue > self.alpha
        
        return {
            "method": self.name,
            "decision": "accept" if accept else "reject",
            "confidence": float(ks_pvalue),
            "time": time.time() - start_time,
            "metadata": {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "t_statistic": float(t_stat),
                "t_pvalue": float(t_pvalue),
                "alpha": self.alpha
            }
        }


class ExternalBaseline(BaselineMethod):
    """Wrapper for external baseline methods"""
    
    def __init__(self, name: str, command: str = None):
        super().__init__(f"external_{name}")
        self.command = command
    
    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        """Call external verification method"""
        start_time = time.time()
        
        # Check if external method is available
        if self.command and os.path.exists(self.command):
            # Would call external command here
            # For now, return not available
            pass
        
        return {
            "method": self.name,
            "decision": "unavailable",
            "confidence": 0.0,
            "time": time.time() - start_time,
            "metadata": {
                "error": "external_method_not_available",
                "command": self.command
            }
        }


class FBIBaseline(BaselineMethod):
    """Benign-input fingerprint baseline"""

    def __init__(self):
        super().__init__("fbi")

    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        start_time = time.time()
        fp1 = benign_input_fingerprint(outputs1)
        fp2 = benign_input_fingerprint(outputs2)
        match = fp1 == fp2
        return {
            "method": self.name,
            "decision": "accept" if match else "reject",
            "confidence": 1.0 if match else 0.0,
            "time": time.time() - start_time,
            "metadata": {"fp1": fp1[:16] + "...", "fp2": fp2[:16] + "..."},
        }


class AdversarialTrajectoryBaseline(BaselineMethod):
    """Adversarial-trajectory fingerprint baseline"""

    def __init__(self):
        super().__init__("adv_traj")

    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        start_time = time.time()
        fp1 = adversarial_trajectory_fingerprint(outputs1)
        fp2 = adversarial_trajectory_fingerprint(outputs2)
        match = fp1 == fp2
        return {
            "method": self.name,
            "decision": "accept" if match else "reject",
            "confidence": 1.0 if match else 0.0,
            "time": time.time() - start_time,
            "metadata": {"fp1": fp1[:16] + "...", "fp2": fp2[:16] + "..."},
        }


class FixedNAggregationBaseline(BaselineMethod):
    """Fixed-n distance aggregation baseline"""

    def __init__(self, n: int = 32, metric: str = "l2", threshold: float = 0.1):
        super().__init__(f"fixedn_{metric}")
        self.n = n
        self.metric = metric
        self.threshold = threshold

    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        start_time = time.time()
        dist = fixed_n_aggregation_distance(outputs1, outputs2, n=self.n, metric=self.metric)
        accept = dist < self.threshold
        return {
            "method": self.name,
            "decision": "accept" if accept else "reject",
            "confidence": float(math.exp(-dist / max(self.threshold, 1e-8))),
            "time": time.time() - start_time,
            "metadata": {"distance": dist, "n": self.n, "metric": self.metric},
        }


class SequentialHoeffdingBaseline(BaselineMethod):
    """Sequential Hoeffding test baseline"""

    def __init__(self, epsilon: float = 0.05):
        super().__init__("seq_hoeffding")
        self.epsilon = epsilon

    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        start_time = time.time()
        decision, n_q = sequential_hoeffding_test(outputs1, outputs2, epsilon=self.epsilon)
        return {
            "method": self.name,
            "decision": decision,
            "confidence": 1.0 if decision == "accept" else 0.0,
            "time": time.time() - start_time,
            "metadata": {"n_queries": n_q, "epsilon": self.epsilon},
        }


class SequentialSPRTBaseline(BaselineMethod):
    """Simple SPRT baseline for mean differences"""

    def __init__(self, alpha: float = 0.05, beta: float = 0.05, mu0: float = 0.0, mu1: float = 0.1, sigma: float = 1.0):
        super().__init__("seq_sprt")
        self.alpha = alpha
        self.beta = beta
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma

    def verify(self, outputs1: List, outputs2: List) -> Dict[str, Any]:
        start_time = time.time()
        arr1 = np.asarray(outputs1, dtype=np.float64).flatten()
        arr2 = np.asarray(outputs2, dtype=np.float64).flatten()

        llr = 0.0
        n_q = 0
        A = math.log((1 - self.beta) / self.alpha)
        B = math.log(self.beta / (1 - self.alpha))
        decision = "accept"

        for o1, o2 in zip(arr1, arr2):
            n_q += 1
            x = o1 - o2
            llr += ((x - self.mu0) ** 2 - (x - self.mu1) ** 2) / (2 * self.sigma ** 2)
            if llr >= A:
                decision = "reject"
                break
            if llr <= B:
                decision = "accept"
                break

        return {
            "method": self.name,
            "decision": decision,
            "confidence": 1.0,
            "time": time.time() - start_time,
            "metadata": {"n_queries": n_q, "llr": llr},
        }


def generate_test_outputs(n_samples: int = 100, 
                         output_dim: int = 10,
                         noise_level: float = 0.1) -> Tuple[List, List, List]:
    """
    Generate test outputs for baseline comparison
    
    Returns:
        genuine1, genuine2 (from same model), impostor (from different model)
    """
    # Base model output
    base = np.random.randn(n_samples, output_dim)
    
    # Genuine: same model with small noise
    genuine1 = base + noise_level * np.random.randn(n_samples, output_dim)
    genuine2 = base + noise_level * np.random.randn(n_samples, output_dim)
    
    # Impostor: different model
    impostor = np.random.randn(n_samples, output_dim) * 2 + 1
    
    return genuine1.tolist(), genuine2.tolist(), impostor.tolist()


def run_baseline_comparison(baselines: List[BaselineMethod],
                           genuine1: List,
                           genuine2: List,
                           impostor: List) -> Dict[str, Any]:
    """
    Run all baselines and compare results
    
    Args:
        baselines: List of baseline methods
        genuine1, genuine2: Outputs from same model
        impostor: Outputs from different model
        
    Returns:
        Comparison results
    """
    results = {
        "n_baselines": len(baselines),
        "n_samples": len(genuine1),
        "genuine_results": [],
        "impostor_results": [],
        "summary": {}
    }
    
    for baseline in baselines:
        print(f"  Running {baseline.name}...")
        
        # Test on genuine pair
        genuine_result = baseline.verify(genuine1, genuine2)
        results["genuine_results"].append(genuine_result)
        
        # Test on impostor pair
        impostor_result = baseline.verify(genuine1, impostor)
        results["impostor_results"].append(impostor_result)
    
    # Compute summary statistics
    for baseline_idx, baseline in enumerate(baselines):
        gen_res = results["genuine_results"][baseline_idx]
        imp_res = results["impostor_results"][baseline_idx]
        
        # Skip if unavailable
        if gen_res["decision"] == "unavailable":
            continue
        
        tp = 1 if gen_res["decision"] == "accept" else 0  # True positive
        tn = 1 if imp_res["decision"] == "reject" else 0  # True negative
        fp = 1 if imp_res["decision"] == "accept" else 0  # False positive
        fn = 1 if gen_res["decision"] == "reject" else 0  # False negative
        
        results["summary"][baseline.name] = {
            "accuracy": (tp + tn) / 2,
            "tpr": tp,  # True positive rate
            "tnr": tn,  # True negative rate
            "fpr": fp,  # False positive rate
            "fnr": fn,  # False negative rate
            "avg_time": (gen_res["time"] + imp_res["time"]) / 2,
            "avg_confidence": (gen_res["confidence"] + imp_res["confidence"]) / 2,
            "avg_queries": (
                gen_res["metadata"].get("n_queries", len(genuine1))
                + imp_res["metadata"].get("n_queries", len(impostor))
            ) / 2,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare baseline verification methods')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--output_dim', type=int, default=10,
                       help='Output dimension')
    parser.add_argument('--output_dir', type=str, default='outputs/baselines',
                       help='Output directory')
    parser.add_argument('--external_ipguard', type=str,
                       help='Path to IPGuard executable (if available)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Baseline Comparison")
    print("=" * 50)
    
    # Initialize baselines
    baselines = [
        NaiveHashBaseline(),
        SimpleDistanceBaseline(threshold=0.1, metric="l2"),
        SimpleDistanceBaseline(threshold=0.2, metric="cosine"),
        SimpleDistanceBaseline(threshold=0.5, metric="l1"),
        StatisticalBaseline(alpha=0.05),
        FBIBaseline(),
        AdversarialTrajectoryBaseline(),
        FixedNAggregationBaseline(n=32, metric="l2"),
        FixedNAggregationBaseline(n=32, metric="hamming"),
        SequentialHoeffdingBaseline(epsilon=0.05),
        SequentialSPRTBaseline(alpha=0.05, beta=0.05),
    ]
    
    # Add external baselines if available
    if args.external_ipguard:
        baselines.append(ExternalBaseline("ipguard", args.external_ipguard))
    
    print(f"Testing {len(baselines)} baseline methods")
    
    # Generate test data
    print(f"\nGenerating test outputs (n={args.n_samples}, dim={args.output_dim})...")
    genuine1, genuine2, impostor = generate_test_outputs(
        args.n_samples, 
        args.output_dim,
        noise_level=0.1
    )
    
    # Run comparison
    print("\nRunning baseline comparison...")
    results = run_baseline_comparison(baselines, genuine1, genuine2, impostor)
    
    # Add metadata
    results["metadata"] = {
        "n_samples": args.n_samples,
        "output_dim": args.output_dim,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    # Save results
    results_file = output_dir / "baseline_comparison.json"
    with open(results_file, 'w') as f:
        safe_json_dump(results, f)
    print(f"\nResults saved to {results_file}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Accuracy':<10} {'TPR':<10} {'FPR':<10} {'Queries':<10}")
    print("-" * 70)

    for method_name, stats in results["summary"].items():
        print(
            f"{method_name:<30} "
            f"{stats['accuracy']:<10.2f} "
            f"{stats['tpr']:<10.2f} "
            f"{stats['fpr']:<10.2f} "
            f"{stats.get('avg_queries', 0):<10.2f}"
        )
    
    # Generate comparison CSV
    csv_file = output_dir / "baseline_comparison.csv"
    with open(csv_file, 'w') as f:
        f.write("method,accuracy,tpr,tnr,fpr,fnr,avg_time_ms,avg_confidence,avg_queries\n")
        for method_name, stats in results["summary"].items():
            f.write(
                f"{method_name},"
                f"{stats['accuracy']},"
                f"{stats['tpr']},"
                f"{stats['tnr']},"
                f"{stats['fpr']},"
                f"{stats['fnr']},"
                f"{stats['avg_time']*1000},"
                f"{stats['avg_confidence']},"
                f"{stats.get('avg_queries', 0)}\n"
            )
    print(f"\nCSV saved to {csv_file}")
    
    # Identify best baseline
    if results["summary"]:
        best_method = max(results["summary"].items(), 
                         key=lambda x: x[1]['accuracy'])
        print(f"\nBest baseline: {best_method[0]} (accuracy={best_method[1]['accuracy']:.2f})")
    
    print("\nComparison complete!")


if __name__ == '__main__':
    main()