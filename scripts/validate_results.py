#!/usr/bin/env python3
"""
Results Validation Script

Compares experimental results against paper claims and generates validation reports.
This script automates the validation process referenced in the Makefile.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate experimental results against paper claims"
    )
    parser.add_argument(
        "--claimed", 
        type=str, 
        required=True,
        help="Path to paper claims file (markdown or JSON)"
    )
    parser.add_argument(
        "--actual", 
        type=str, 
        required=True,
        help="Path to experimental results directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output file for validation report"
    )
    parser.add_argument(
        "--format", 
        choices=["html", "json", "markdown"], 
        default="html",
        help="Output format"
    )
    parser.add_argument(
        "--tolerance", 
        type=float, 
        default=0.1,
        help="Tolerance for metric comparisons (default: 0.1 = 10%)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def extract_paper_claims(claims_file: Path) -> Dict[str, Any]:
    """Extract paper claims from markdown or JSON file."""
    logger.info(f"Extracting paper claims from {claims_file}")
    
    # Default paper claims based on PoT paper
    default_claims = {
        "far": 0.01,
        "frr": 0.01,
        "accuracy": 0.99,
        "efficiency_gain": 0.90,
        "average_queries": 10.0,
        "confidence_level": 0.95
    }
    
    if not claims_file.exists():
        logger.warning(f"Claims file not found: {claims_file}")
        logger.info("Using default PoT paper claims")
        return default_claims
    
    if claims_file.suffix.lower() == '.json':
        try:
            with open(claims_file, 'r') as f:
                claims = json.load(f)
                return claims
        except Exception as e:
            logger.error(f"Failed to parse JSON claims file: {e}")
            return default_claims
    
    # For markdown files, extract key metrics
    try:
        with open(claims_file, 'r') as f:
            content = f.read()
            
        # Simple extraction - look for key patterns
        claims = default_claims.copy()
        
        # Look for patterns like "FAR: 1%" or "False Accept Rate: 0.01"
        import re
        
        # FAR patterns
        far_pattern = r'(?:FAR|False Accept Rate).*?(\d+(?:\.\d+)?)\s*%?'
        far_match = re.search(far_pattern, content, re.IGNORECASE)
        if far_match:
            far_value = float(far_match.group(1))
            if far_value > 1:  # Assume percentage
                far_value /= 100
            claims["far"] = far_value
        
        # FRR patterns
        frr_pattern = r'(?:FRR|False Reject Rate).*?(\d+(?:\.\d+)?)\s*%?'
        frr_match = re.search(frr_pattern, content, re.IGNORECASE)
        if frr_match:
            frr_value = float(frr_match.group(1))
            if frr_value > 1:  # Assume percentage
                frr_value /= 100
            claims["frr"] = frr_value
        
        # Accuracy patterns
        acc_pattern = r'(?:Accuracy|Success Rate).*?(\d+(?:\.\d+)?)\s*%?'
        acc_match = re.search(acc_pattern, content, re.IGNORECASE)
        if acc_match:
            acc_value = float(acc_match.group(1))
            if acc_value > 1:  # Assume percentage
                acc_value /= 100
            claims["accuracy"] = acc_value
        
        logger.info(f"Extracted claims: {claims}")
        return claims
        
    except Exception as e:
        logger.error(f"Failed to parse markdown claims file: {e}")
        return default_claims

def collect_experimental_results(results_dir: Path) -> Dict[str, Any]:
    """Collect experimental results from directory."""
    logger.info(f"Collecting experimental results from {results_dir}")
    
    results = {
        "metrics": {},
        "raw_data": [],
        "metadata": {}
    }
    
    # Look for metrics files
    for metrics_file in results_dir.glob("**/test_metrics.json"):
        logger.info(f"Found test metrics: {metrics_file}")
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                results["metrics"].update(data)
        except Exception as e:
            logger.warning(f"Failed to load {metrics_file}: {e}")
    
    # Look for other metrics files
    for metrics_file in results_dir.glob("**/metrics_*.json"):
        logger.info(f"Found metrics file: {metrics_file}")
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    results["metrics"].update(data)
        except Exception as e:
            logger.warning(f"Failed to load {metrics_file}: {e}")
    
    # Look for raw results
    for result_file in results_dir.glob("**/results_*.json"):
        logger.info(f"Found raw results: {result_file}")
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                results["raw_data"].append(data)
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")
    
    # Extract metrics from raw data if no metrics files found
    if not results["metrics"] and results["raw_data"]:
        logger.info("Extracting metrics from raw data")
        for raw_data in results["raw_data"]:
            if "metrics" in raw_data:
                results["metrics"].update(raw_data["metrics"])
    
    logger.info(f"Collected metrics: {list(results['metrics'].keys())}")
    return results

def compare_metrics(claimed: Dict[str, Any], actual: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """Compare claimed vs actual metrics."""
    logger.info("Comparing metrics with tolerance of {:.1%}".format(tolerance))
    
    comparison = {
        "total_metrics": 0,
        "matched_metrics": 0,
        "missing_metrics": [],
        "failed_metrics": [],
        "passed_metrics": [],
        "details": {}
    }
    
    for metric_name, claimed_value in claimed.items():
        comparison["total_metrics"] += 1
        
        # Try to find corresponding actual value
        actual_value = None
        
        # Direct match
        if metric_name in actual:
            actual_value = actual[metric_name]
        # Try variations
        elif metric_name.upper() in actual:
            actual_value = actual[metric_name.upper()]
        elif metric_name.lower() in actual:
            actual_value = actual[metric_name.lower()]
        # Try with underscores
        elif metric_name.replace("_", "") in actual:
            actual_value = actual[metric_name.replace("_", "")]
        
        if actual_value is None:
            comparison["missing_metrics"].append(metric_name)
            comparison["details"][metric_name] = {
                "status": "missing",
                "claimed": claimed_value,
                "actual": None,
                "difference": None,
                "relative_difference": None
            }
            continue
        
        # Compare values
        try:
            claimed_val = float(claimed_value)
            actual_val = float(actual_value)
            
            difference = actual_val - claimed_val
            relative_difference = abs(difference) / abs(claimed_val) if claimed_val != 0 else float('inf')
            
            status = "passed" if relative_difference <= tolerance else "failed"
            
            comparison["details"][metric_name] = {
                "status": status,
                "claimed": claimed_val,
                "actual": actual_val,
                "difference": difference,
                "relative_difference": relative_difference
            }
            
            if status == "passed":
                comparison["passed_metrics"].append(metric_name)
                comparison["matched_metrics"] += 1
            else:
                comparison["failed_metrics"].append(metric_name)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not compare {metric_name}: {e}")
            comparison["details"][metric_name] = {
                "status": "error",
                "claimed": claimed_value,
                "actual": actual_value,
                "error": str(e)
            }
    
    # Calculate overall success rate
    comparison["success_rate"] = comparison["matched_metrics"] / comparison["total_metrics"] if comparison["total_metrics"] > 0 else 0
    
    return comparison

def generate_html_report(comparison: Dict[str, Any], claimed: Dict[str, Any], 
                        actual: Dict[str, Any], output_file: Path, tolerance: float):
    """Generate HTML validation report."""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PoT Experiment Validation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ 
            background-color: #ecf0f1; 
            padding: 20px; 
            border-radius: 5px; 
            margin: 20px 0; 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric-card {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .success-rate {{
            font-size: 2em;
            font-weight: bold;
            color: {('#27ae60' if comparison['success_rate'] >= 0.8 else '#e74c3c' if comparison['success_rate'] < 0.5 else '#f39c12')};
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metric-table th, .metric-table td {{
            border: 1px solid #bdc3c7;
            padding: 12px;
            text-align: left;
        }}
        .metric-table th {{
            background-color: #34495e;
            color: white;
        }}
        .status-passed {{ background-color: #d5f4e6; color: #27ae60; font-weight: bold; }}
        .status-failed {{ background-color: #fadbd8; color: #e74c3c; font-weight: bold; }}
        .status-missing {{ background-color: #fef9e7; color: #f39c12; font-weight: bold; }}
        .status-error {{ background-color: #f4f4f4; color: #7f8c8d; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 PoT Experiment Validation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Tolerance:</strong> {tolerance:.1%}</p>
        
        <div class="summary">
            <div class="metric-card">
                <h3>Overall Success Rate</h3>
                <div class="success-rate">{comparison['success_rate']:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Total Metrics</h3>
                <div style="font-size: 1.5em; font-weight: bold;">{comparison['total_metrics']}</div>
            </div>
            <div class="metric-card">
                <h3>Passed</h3>
                <div style="font-size: 1.5em; color: #27ae60; font-weight: bold;">{comparison['matched_metrics']}</div>
            </div>
            <div class="metric-card">
                <h3>Failed</h3>
                <div style="font-size: 1.5em; color: #e74c3c; font-weight: bold;">{len(comparison['failed_metrics'])}</div>
            </div>
            <div class="metric-card">
                <h3>Missing</h3>
                <div style="font-size: 1.5em; color: #f39c12; font-weight: bold;">{len(comparison['missing_metrics'])}</div>
            </div>
        </div>
        
        <h2>📊 Detailed Comparison</h2>
        <table class="metric-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Status</th>
                    <th>Claimed</th>
                    <th>Actual</th>
                    <th>Difference</th>
                    <th>Relative Diff</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for metric_name, details in comparison["details"].items():
        status_class = f"status-{details['status']}"
        status_text = details['status'].upper()
        
        claimed_val = details.get('claimed', 'N/A')
        actual_val = details.get('actual', 'N/A')
        difference = details.get('difference', 'N/A')
        rel_diff = details.get('relative_difference', 'N/A')
        
        if isinstance(claimed_val, (int, float)):
            claimed_val = f"{claimed_val:.4f}"
        if isinstance(actual_val, (int, float)):
            actual_val = f"{actual_val:.4f}"
        if isinstance(difference, (int, float)):
            difference = f"{difference:+.4f}"
        if isinstance(rel_diff, (int, float)):
            rel_diff = f"{rel_diff:.1%}"
        
        html_content += f"""
                <tr>
                    <td><strong>{metric_name}</strong></td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{claimed_val}</td>
                    <td>{actual_val}</td>
                    <td>{difference}</td>
                    <td>{rel_diff}</td>
                </tr>"""
    
    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML validation report: {output_file}")

def generate_json_report(comparison: Dict[str, Any], claimed: Dict[str, Any], 
                        actual: Dict[str, Any], output_file: Path, tolerance: float):
    """Generate JSON validation report."""
    
    report = {
        "validation_report": {
            "timestamp": datetime.now().isoformat(),
            "tolerance": tolerance,
            "summary": {
                "success_rate": comparison["success_rate"],
                "total_metrics": comparison["total_metrics"],
                "passed_metrics": comparison["matched_metrics"],
                "failed_metrics": len(comparison["failed_metrics"]),
                "missing_metrics": len(comparison["missing_metrics"])
            },
            "comparison": comparison,
            "claimed_values": claimed,
            "actual_values": actual
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Generated JSON validation report: {output_file}")

def main():
    """Main function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting validation process")
    logger.info(f"Claims file: {args.claimed}")
    logger.info(f"Results directory: {args.actual}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Tolerance: {args.tolerance:.1%}")
    
    # Extract paper claims
    claims_file = Path(args.claimed)
    claimed_metrics = extract_paper_claims(claims_file)
    
    # Collect experimental results
    results_dir = Path(args.actual)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        sys.exit(1)
    
    experimental_results = collect_experimental_results(results_dir)
    actual_metrics = experimental_results["metrics"]
    
    if not actual_metrics:
        logger.error("No metrics found in experimental results")
        sys.exit(1)
    
    # Compare metrics
    comparison = compare_metrics(claimed_metrics, actual_metrics, args.tolerance)
    
    # Generate report
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "html":
        generate_html_report(comparison, claimed_metrics, actual_metrics, output_file, args.tolerance)
    elif args.format == "json":
        generate_json_report(comparison, claimed_metrics, actual_metrics, output_file, args.tolerance)
    elif args.format == "markdown":
        # Simple markdown for now
        with open(output_file, 'w') as f:
            f.write(f"# Validation Report\\n\\n")
            f.write(f"**Success Rate:** {comparison['success_rate']:.1%}\\n")
            f.write(f"**Passed:** {comparison['matched_metrics']}/{comparison['total_metrics']}\\n")
            f.write(f"**Failed:** {len(comparison['failed_metrics'])}\\n")
            f.write(f"**Missing:** {len(comparison['missing_metrics'])}\\n")
        logger.info(f"Generated Markdown validation report: {output_file}")
    
    # Print summary
    logger.info("Validation completed!")
    logger.info(f"Success rate: {comparison['success_rate']:.1%}")
    logger.info(f"Passed: {comparison['matched_metrics']}/{comparison['total_metrics']}")
    
    if comparison['failed_metrics']:
        logger.warning(f"Failed metrics: {comparison['failed_metrics']}")
    
    if comparison['missing_metrics']:
        logger.warning(f"Missing metrics: {comparison['missing_metrics']}")
    
    # Exit with appropriate code
    if comparison['success_rate'] >= 0.8:
        logger.info("✅ Validation PASSED")
        sys.exit(0)
    else:
        logger.error("❌ Validation FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()