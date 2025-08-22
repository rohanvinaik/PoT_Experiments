#!/usr/bin/env python3
"""
Example usage of the ReportGenerator for PoT experiments.

This demonstrates how to integrate the automated report generator
with experimental results from the PoT framework.
"""

import json
from pathlib import Path
from pot.experiments.report_generator import ReportGenerator, create_sample_results


def example_basic_usage():
    """Basic usage example with sample data."""
    print("=== Basic ReportGenerator Usage ===")
    
    # Create sample results
    sample_file = "example_results.json"
    create_sample_results(sample_file)
    print(f"Created sample results: {sample_file}")
    
    # Initialize report generator
    generator = ReportGenerator(sample_file)
    print(f"Loaded {len(generator.data)} experimental results")
    
    # Generate individual reports
    print("\nGenerating individual reports...")
    
    # Markdown report
    markdown = generator.generate_markdown_report()
    print(f"  Markdown report: {len(markdown)} characters")
    
    # LaTeX tables
    latex = generator.generate_latex_tables()
    print(f"  LaTeX tables: {len(latex)} characters")
    
    # HTML report
    html = generator.generate_html_report()
    print(f"  HTML report: {len(html)} characters")
    
    # JSON export
    json_data = generator.generate_json_export()
    print(f"  JSON export: {len(json_data)} characters")
    
    # Visualizations
    plots = generator.generate_plots()
    print(f"  Generated {len(plots)} plots")
    
    print(f"\nAll reports saved to: {generator.output_dir}")


def example_comprehensive_suite():
    """Example of generating the complete report suite."""
    print("\n=== Comprehensive Report Suite ===")
    
    # Create sample results with different challenge families
    extended_results = [
        {
            "experiment_id": "exp_vision_freq_001",
            "far": 0.008,
            "frr": 0.012,
            "accuracy": 0.994,
            "queries": 8,
            "processing_time": 0.234,
            "challenge_family": "vision:freq",
            "threshold": 0.05,
            "timestamp": "2024-01-15T10:30:00"
        },
        {
            "experiment_id": "exp_vision_freq_002",
            "far": 0.006,
            "frr": 0.015,
            "accuracy": 0.991,
            "queries": 12,
            "processing_time": 0.287,
            "challenge_family": "vision:freq",
            "threshold": 0.03,
            "timestamp": "2024-01-15T10:35:00"
        },
        {
            "experiment_id": "exp_vision_texture_001",
            "far": 0.011,
            "frr": 0.009,
            "accuracy": 0.996,
            "queries": 6,
            "processing_time": 0.198,
            "challenge_family": "vision:texture",
            "threshold": 0.05,
            "timestamp": "2024-01-15T10:40:00"
        },
        {
            "experiment_id": "exp_lm_templates_001",
            "far": 0.009,
            "frr": 0.011,
            "accuracy": 0.995,
            "queries": 7,
            "processing_time": 0.267,
            "challenge_family": "lm:templates",
            "threshold": 0.04,
            "timestamp": "2024-01-15T10:45:00"
        },
        {
            "experiment_id": "exp_lm_templates_002",
            "far": 0.007,
            "frr": 0.013,
            "accuracy": 0.993,
            "queries": 9,
            "processing_time": 0.312,
            "challenge_family": "lm:templates",
            "threshold": 0.05,
            "timestamp": "2024-01-15T10:50:00"
        }
    ]
    
    sample_file = "extended_results.json"
    with open(sample_file, 'w') as f:
        json.dump(extended_results, f, indent=2)
    
    # Create custom paper claims
    custom_claims = {
        "far": 0.01,
        "frr": 0.01,
        "accuracy": 0.99,
        "efficiency_gain": 0.85,
        "average_queries": 8.0,
        "confidence_level": 0.95
    }
    
    claims_file = "custom_claims.json"
    with open(claims_file, 'w') as f:
        json.dump(custom_claims, f, indent=2)
    
    # Initialize with custom claims
    generator = ReportGenerator(sample_file, claims_file)
    
    # Generate complete report suite
    reports = generator.generate_all_reports()
    
    print(f"Generated {len(reports)} report files:")
    for report_type, path in reports.items():
        print(f"  {report_type}: {Path(path).name}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total Experiments: {len(generator.data)}")
    print(f"  FAR: {generator.metrics.far:.4f}")
    print(f"  FRR: {generator.metrics.frr:.4f}")
    print(f"  Accuracy: {generator.metrics.accuracy:.4f}")
    print(f"  Avg Queries: {generator.metrics.avg_queries:.1f}")
    print(f"  Discrepancies: {len(generator.discrepancies)}")
    
    # Show discrepancies
    if generator.discrepancies:
        print(f"\nIdentified Discrepancies:")
        for d in generator.discrepancies:
            print(f"  {d.metric.upper()}: {d.relative_difference:.1%} ({d.severity})")
    
    print(f"\nOpen {generator.output_dir}/report.html for interactive viewing")


def example_integration_with_experiments():
    """Example showing integration with existing PoT experiment modules."""
    print("\n=== Integration with PoT Experiments ===")
    
    # Simulate results from pot.experiments.reproducible_runner
    simulated_experiment_results = {
        "experiment_config": {
            "experiment_name": "reproducible_test",
            "model_type": "vision",
            "challenge_families": ["vision:freq", "vision:texture"],
            "n_challenges_per_family": 10,
            "alpha": 0.05,
            "beta": 0.05
        },
        "verification_results": [
            {
                "challenge_id": "ch_001",
                "verified": True,
                "ground_truth": True,
                "stopping_time": 8,
                "p_value": 0.03,
                "challenge_family": "vision:freq"
            },
            {
                "challenge_id": "ch_002",
                "verified": False,
                "ground_truth": False,
                "stopping_time": 12,
                "p_value": 0.02,
                "challenge_family": "vision:texture"
            }
        ],
        "metrics_summary": {
            "far": 0.009,
            "frr": 0.011,
            "accuracy": 0.995,
            "total_queries": 20,
            "avg_stopping_time": 10.0,
            "processing_time": 0.453
        }
    }
    
    # Convert to ReportGenerator format
    report_data = []
    config = simulated_experiment_results["experiment_config"]
    metrics = simulated_experiment_results["metrics_summary"]
    
    # Create a summary record
    report_record = {
        "experiment_name": config["experiment_name"],
        "model_type": config["model_type"],
        "far": metrics["far"],
        "frr": metrics["frr"],
        "accuracy": metrics["accuracy"],
        "queries": metrics["total_queries"],
        "avg_stopping_time": metrics["avg_stopping_time"],
        "processing_time": metrics["processing_time"],
        "alpha": config["alpha"],
        "beta": config["beta"],
        "timestamp": "2024-01-15T11:00:00"
    }
    
    # Add per-challenge records
    for result in simulated_experiment_results["verification_results"]:
        challenge_record = {
            "challenge_id": result["challenge_id"],
            "challenge_family": result["challenge_family"],
            "verified": result["verified"],
            "ground_truth": result["ground_truth"],
            "queries": result["stopping_time"],
            "p_value": result["p_value"],
            "processing_time": 0.2,  # Simulated
            "timestamp": "2024-01-15T11:00:00"
        }
        report_data.append(challenge_record)
    
    # Add summary record
    report_data.append(report_record)
    
    # Save and generate reports
    integration_file = "integration_results.json"
    with open(integration_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    generator = ReportGenerator(integration_file)
    reports = generator.generate_all_reports()
    
    print(f"Generated integrated reports in: {generator.output_dir}")
    print(f"Processed {len(generator.data)} records from PoT experiments")


def example_custom_analysis():
    """Example of custom analysis and extensions."""
    print("\n=== Custom Analysis Extensions ===")
    
    # Create results with custom metrics
    custom_results = [
        {
            "experiment_id": "custom_001",
            "far": 0.008,
            "frr": 0.012,
            "accuracy": 0.994,
            "queries": 8,
            "processing_time": 0.234,
            "challenge_family": "vision:freq",
            "custom_metric_1": 0.85,  # Custom efficiency metric
            "custom_metric_2": 42.7,  # Custom complexity score
            "model_size_mb": 15.2,
            "gpu_memory_mb": 512,
            "timestamp": "2024-01-15T10:30:00"
        },
        {
            "experiment_id": "custom_002",
            "far": 0.006,
            "frr": 0.015,
            "accuracy": 0.991,
            "queries": 12,
            "processing_time": 0.287,
            "challenge_family": "vision:texture",
            "custom_metric_1": 0.78,
            "custom_metric_2": 38.9,
            "model_size_mb": 18.7,
            "gpu_memory_mb": 768,
            "timestamp": "2024-01-15T10:35:00"
        }
    ]
    
    custom_file = "custom_results.json"
    with open(custom_file, 'w') as f:
        json.dump(custom_results, f, indent=2)
    
    generator = ReportGenerator(custom_file)
    
    # Access raw data for custom analysis
    print("Custom Analysis Results:")
    for result in generator.data:
        if 'custom_metric_1' in result:
            print(f"  {result['experiment_id']}: Efficiency={result['custom_metric_1']:.2f}, "
                  f"Complexity={result['custom_metric_2']:.1f}")
    
    # Generate reports with custom data included
    reports = generator.generate_all_reports()
    print(f"Custom analysis reports saved to: {generator.output_dir}")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_comprehensive_suite()
    example_integration_with_experiments()
    example_custom_analysis()
    
    print("\n" + "="*60)
    print("Examples completed! Check the generated report directories.")
    print("Open any report.html file in your browser for best viewing.")
    print("="*60)