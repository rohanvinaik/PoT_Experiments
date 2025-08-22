#!/usr/bin/env python3
"""
Standardized reporting module for PoT framework.
Provides clean, focused reporting with essential metrics only.
"""

import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path


class ReportFormatter:
    """Clean, focused report formatting for PoT experiments."""
    
    @staticmethod
    def format_statistical_identity(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format statistical identity test results.
        
        Expected input keys:
        - alpha, beta: error rates
        - n_used: number of queries used
        - distances: array of distances
        - decision: H0/H1/continue
        - timing: dict with load, inference times
        - positions_per_prompt: K value
        """
        distances = result.get('distances', [])
        mean_dist = np.mean(distances) if distances else 0
        
        # Calculate 99% confidence interval
        if len(distances) > 1:
            std_err = np.std(distances) / np.sqrt(len(distances))
            z_99 = 2.576  # 99% confidence
            ci_lo = mean_dist - z_99 * std_err
            ci_hi = mean_dist + z_99 * std_err
            half_width = z_99 * std_err
            rel_me = (half_width / mean_dist * 100) if mean_dist > 0 else 0
        else:
            ci_lo = ci_hi = mean_dist
            half_width = 0
            rel_me = 0
        
        # Map decision to standardized format
        decision_map = {
            'H0': 'SAME',
            'H1': 'DIFFERENT', 
            'continue': 'UNDECIDED',
            'accept': 'SAME',
            'reject': 'DIFFERENT'
        }
        raw_decision = result.get('decision', 'UNDECIDED')
        decision = decision_map.get(raw_decision, raw_decision.upper())
        
        return {
            "alpha": result.get('alpha', 0.01),
            "beta": result.get('beta', 0.01),
            "n_used": result.get('n_used', len(distances)),
            "mean": round(mean_dist, 6),
            "ci_99": [round(ci_lo, 6), round(ci_hi, 6)],
            "half_width": round(half_width, 6),
            "rel_me": round(rel_me, 2),
            "decision": decision,
            "positions_per_prompt": result.get('positions_per_prompt', 1),
            "time": {
                "load": round(result.get('timing', {}).get('load', 0), 3),
                "infer_total": round(result.get('timing', {}).get('inference', 0), 3),
                "per_query": round(result.get('timing', {}).get('per_query', 0), 6)
            }
        }
    
    @staticmethod
    def format_fuzzy_hash(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format fuzzy hash verification results.
        
        Expected input keys:
        - algorithm: TLSH/SSDEEP/SHA256
        - threshold: similarity threshold
        - similarities: list of similarity scores
        - verified: boolean or list of booleans
        """
        similarities = result.get('similarities', [])
        if isinstance(result.get('verified'), list):
            pass_rate = sum(result['verified']) / len(result['verified']) if result['verified'] else 0
        else:
            pass_rate = 1.0 if result.get('verified') else 0.0
        
        # Get example scores (first 5)
        example_scores = similarities[:5] if similarities else []
        
        return {
            "algorithm": result.get('algorithm', 'TLSH').upper(),
            "threshold": result.get('threshold', 0.85),
            "pass_rate": round(pass_rate, 3),
            "example_scores": [round(s, 3) for s in example_scores],
            "mean_similarity": round(np.mean(similarities), 3) if similarities else 0,
            "min_similarity": round(min(similarities), 3) if similarities else 0,
            "max_similarity": round(max(similarities), 3) if similarities else 0
        }
    
    @staticmethod
    def format_provenance(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format training provenance audit results.
        
        Expected input keys:
        - merkle_root: hex string
        - inclusion_proof: dict with leaf, path, verified
        - compression_stats: dict with original, compressed, ratio
        - checks_passed: list of check names
        """
        return {
            "signed_merkle_root": result.get('merkle_root', 'N/A'),
            "verified_inclusion_proof": {
                "event_hash": result.get('inclusion_proof', {}).get('leaf', 'N/A')[:16] + "...",
                "path_length": len(result.get('inclusion_proof', {}).get('path', [])),
                "verified": result.get('inclusion_proof', {}).get('verified', False)
            },
            "compression_stats": {
                "original_events": result.get('compression_stats', {}).get('original', 0),
                "compressed_events": result.get('compression_stats', {}).get('compressed', 0),
                "compression_ratio": round(result.get('compression_stats', {}).get('ratio', 0), 2)
            },
            "checks_passed": result.get('checks_passed', [])
        }
    
    @staticmethod
    def format_summary(results: Dict[str, Any]) -> str:
        """Generate a clean summary report."""
        lines = []
        lines.append("=" * 70)
        lines.append("POT VERIFICATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append("")
        
        # Statistical Identity Results
        if 'statistical' in results:
            stat = results['statistical']
            lines.append("STATISTICAL IDENTITY VERIFICATION")
            lines.append("-" * 35)
            lines.append(f"Decision: {stat['decision']}")
            lines.append(f"Queries Used: {stat['n_used']}")
            lines.append(f"Mean Distance: {stat['mean']:.6f}")
            lines.append(f"99% CI: [{stat['ci_99'][0]:.6f}, {stat['ci_99'][1]:.6f}]")
            lines.append(f"Relative Margin of Error: {stat['rel_me']:.2f}%")
            lines.append(f"Time per Query: {stat['time']['per_query']:.6f}s")
            lines.append("")
        
        # Fuzzy Hash Results
        if 'fuzzy' in results:
            fuzzy = results['fuzzy']
            lines.append("FUZZY HASH VERIFICATION")
            lines.append("-" * 35)
            lines.append(f"Algorithm: {fuzzy['algorithm']}")
            lines.append(f"Threshold: {fuzzy['threshold']}")
            lines.append(f"Pass Rate: {fuzzy['pass_rate']:.1%}")
            lines.append(f"Example Scores: {fuzzy['example_scores']}")
            lines.append(f"Mean Similarity: {fuzzy['mean_similarity']:.3f}")
            lines.append("")
        
        # Provenance Results
        if 'provenance' in results:
            prov = results['provenance']
            lines.append("TRAINING PROVENANCE AUDIT")
            lines.append("-" * 35)
            lines.append(f"Merkle Root: {prov['signed_merkle_root'][:32]}...")
            lines.append(f"Inclusion Proof: {'✓' if prov['verified_inclusion_proof']['verified'] else '✗'}")
            lines.append(f"Compression: {prov['compression_stats']['original_events']} → "
                        f"{prov['compression_stats']['compressed_events']} "
                        f"(ratio: {prov['compression_stats']['compression_ratio']}x)")
            lines.append(f"Checks Passed: {len(prov['checks_passed'])}/{len(prov['checks_passed'])}")
            for check in prov['checks_passed'][:5]:  # Show first 5
                lines.append(f"  ✓ {check}")
            lines.append("")
        
        # Overall Status
        lines.append("OVERALL STATUS")
        lines.append("-" * 35)
        
        all_passed = True
        if 'statistical' in results:
            stat_pass = results['statistical']['decision'] in ['SAME', 'DIFFERENT']
            lines.append(f"Statistical: {'✓ PASS' if stat_pass else '✗ FAIL'}")
            all_passed &= stat_pass
        
        if 'fuzzy' in results:
            fuzzy_pass = results['fuzzy']['pass_rate'] > 0.95
            lines.append(f"Fuzzy Hash: {'✓ PASS' if fuzzy_pass else '✗ FAIL'}")
            all_passed &= fuzzy_pass
        
        if 'provenance' in results:
            prov_pass = results['provenance']['verified_inclusion_proof']['verified']
            lines.append(f"Provenance: {'✓ PASS' if prov_pass else '✗ FAIL'}")
            all_passed &= prov_pass
        
        lines.append("")
        lines.append(f"VERIFICATION: {'✓ SUCCESSFUL' if all_passed else '✗ FAILED'}")
        lines.append("=" * 70)
        
        return "\n".join(lines)


class ExperimentReporter:
    """Focused experiment reporter with essential metrics only."""
    
    def __init__(self, output_dir: str = "experimental_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.formatter = ReportFormatter()
        self.results = {}
    
    def add_statistical_result(self, result: Dict[str, Any]):
        """Add statistical identity verification result."""
        self.results['statistical'] = self.formatter.format_statistical_identity(result)
    
    def add_fuzzy_result(self, result: Dict[str, Any]):
        """Add fuzzy hash verification result."""
        self.results['fuzzy'] = self.formatter.format_fuzzy_hash(result)
    
    def add_provenance_result(self, result: Dict[str, Any]):
        """Add provenance audit result."""
        self.results['provenance'] = self.formatter.format_provenance(result)
    
    def save_json(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            filename = f"pot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return filepath
    
    def save_text(self, filename: Optional[str] = None):
        """Save results to text file."""
        if filename is None:
            filename = f"pot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(self.formatter.format_summary(self.results))
        
        return filepath
    
    def print_summary(self):
        """Print summary to console."""
        print(self.formatter.format_summary(self.results))
    
    def get_results(self) -> Dict[str, Any]:
        """Get formatted results dictionary."""
        return self.results.copy()


def create_mock_statistical_result() -> Dict[str, Any]:
    """Create a mock statistical result for testing."""
    return {
        'alpha': 0.01,
        'beta': 0.01,
        'n_used': 32,
        'distances': np.random.uniform(0.1, 0.3, 32).tolist(),
        'decision': 'H0',
        'positions_per_prompt': 1,
        'timing': {
            'load': 0.523,
            'inference': 1.234,
            'per_query': 0.0385
        }
    }


def create_mock_fuzzy_result() -> Dict[str, Any]:
    """Create a mock fuzzy result for testing."""
    return {
        'algorithm': 'TLSH',
        'threshold': 0.85,
        'similarities': [0.91, 0.89, 0.93, 0.88, 0.92, 0.90, 0.87],
        'verified': [True, True, True, True, True, True, True]
    }


def create_mock_provenance_result() -> Dict[str, Any]:
    """Create a mock provenance result for testing."""
    return {
        'merkle_root': 'a1b2c3d4e5f6789012345678901234567890123456789012345678901234567890',
        'inclusion_proof': {
            'leaf': 'event_hash_123456789abcdef0123456789abcdef',
            'path': [('left', 'hash1'), ('right', 'hash2'), ('left', 'hash3')],
            'verified': True
        },
        'compression_stats': {
            'original': 1000,
            'compressed': 100,
            'ratio': 10.0
        },
        'checks_passed': [
            'Merkle tree consistency',
            'Event ordering',
            'Timestamp monotonicity',
            'Signature verification',
            'Hash chain integrity'
        ]
    }


if __name__ == "__main__":
    # Test the reporter
    reporter = ExperimentReporter()
    
    # Add mock results
    reporter.add_statistical_result(create_mock_statistical_result())
    reporter.add_fuzzy_result(create_mock_fuzzy_result())
    reporter.add_provenance_result(create_mock_provenance_result())
    
    # Print and save
    reporter.print_summary()
    json_path = reporter.save_json()
    text_path = reporter.save_text()
    
    print(f"\nReports saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {text_path}")