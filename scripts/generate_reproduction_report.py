#!/usr/bin/env python3
"""
Reproduction Report Generator

Generates comprehensive reports from PoT experiment results including:
- HTML reports with interactive visualizations
- PDF reports for academic use
- JSON summaries for programmatic access
- Markdown reports for documentation
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
        description="Generate comprehensive reproduction reports from PoT experiment results"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Input directory containing experiment results"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output directory for generated reports"
    )
    parser.add_argument(
        "--format", 
        choices=["html", "pdf", "json", "markdown", "all"], 
        default="all",
        help="Report format to generate"
    )
    parser.add_argument(
        "--template", 
        choices=["minimal", "standard", "full", "paper"], 
        default="standard",
        help="Report template to use"
    )
    parser.add_argument(
        "--title", 
        type=str, 
        default="PoT Experiment Reproduction Report",
        help="Report title"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def collect_results(input_dir: Path) -> Dict[str, Any]:
    """Collect all experimental results from input directory."""
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_directory": str(input_dir),
            "collected_files": []
        },
        "metrics": {},
        "sequential_analysis": {},
        "raw_results": [],
        "configurations": [],
        "logs": []
    }
    
    logger.info(f"Collecting results from {input_dir}")
    
    # Collect metrics reports
    for metrics_file in input_dir.glob("**/metrics_*.json"):
        logger.info(f"Found metrics file: {metrics_file}")
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                results["metrics"][metrics_file.name] = metrics_data
                results["metadata"]["collected_files"].append(str(metrics_file))
        except Exception as e:
            logger.warning(f"Failed to load metrics file {metrics_file}: {e}")
    
    # Collect sequential analysis
    for seq_file in input_dir.glob("**/sequential_*.json"):
        logger.info(f"Found sequential analysis: {seq_file}")
        try:
            with open(seq_file, 'r') as f:
                seq_data = json.load(f)
                results["sequential_analysis"][seq_file.name] = seq_data
                results["metadata"]["collected_files"].append(str(seq_file))
        except Exception as e:
            logger.warning(f"Failed to load sequential file {seq_file}: {e}")
    
    # Collect raw results
    for result_file in input_dir.glob("**/results_*.json"):
        logger.info(f"Found raw results: {result_file}")
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                results["raw_results"].append({
                    "file": str(result_file),
                    "data": result_data
                })
                results["metadata"]["collected_files"].append(str(result_file))
        except Exception as e:
            logger.warning(f"Failed to load result file {result_file}: {e}")
    
    # Collect configurations
    for config_file in input_dir.glob("**/config.json"):
        logger.info(f"Found configuration: {config_file}")
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                results["configurations"].append({
                    "file": str(config_file),
                    "data": config_data
                })
                results["metadata"]["collected_files"].append(str(config_file))
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
    
    # Collect log files
    for log_file in input_dir.glob("**/*.log"):
        logger.info(f"Found log file: {log_file}")
        results["logs"].append(str(log_file))
        results["metadata"]["collected_files"].append(str(log_file))
    
    logger.info(f"Collected {len(results['metadata']['collected_files'])} files")
    return results

def generate_json_report(results: Dict[str, Any], output_dir: Path, title: str) -> Path:
    """Generate JSON summary report."""
    output_file = output_dir / "reproduction_summary.json"
    
    summary = {
        "title": title,
        "generation_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files_processed": len(results["metadata"]["collected_files"]),
            "metrics_files": len(results["metrics"]),
            "sequential_files": len(results["sequential_analysis"]),
            "result_files": len(results["raw_results"]),
            "config_files": len(results["configurations"]),
            "log_files": len(results["logs"])
        },
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Generated JSON report: {output_file}")
    return output_file

def generate_markdown_report(results: Dict[str, Any], output_dir: Path, title: str) -> Path:
    """Generate Markdown report."""
    output_file = output_dir / "reproduction_report.md"
    
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Input Directory**: {results['metadata']['input_directory']}")
    lines.append("")
    
    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Files Processed**: {len(results['metadata']['collected_files'])}")
    lines.append(f"- **Metrics Files**: {len(results['metrics'])}")
    lines.append(f"- **Sequential Analysis Files**: {len(results['sequential_analysis'])}")
    lines.append(f"- **Raw Result Files**: {len(results['raw_results'])}")
    lines.append(f"- **Configuration Files**: {len(results['configurations'])}")
    lines.append(f"- **Log Files**: {len(results['logs'])}")
    lines.append("")
    
    # Metrics section
    if results["metrics"]:
        lines.append("## Metrics Results")
        lines.append("")
        for filename, metrics_data in results["metrics"].items():
            lines.append(f"### {filename}")
            lines.append("")
            if isinstance(metrics_data, dict):
                for metric_name, value in metrics_data.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"- **{metric_name}**: {value:.4f}")
                    else:
                        lines.append(f"- **{metric_name}**: {value}")
            lines.append("")
    
    # Sequential Analysis section
    if results["sequential_analysis"]:
        lines.append("## Sequential Analysis")
        lines.append("")
        for filename, seq_data in results["sequential_analysis"].items():
            lines.append(f"### {filename}")
            lines.append("")
            if isinstance(seq_data, dict):
                for key, value in seq_data.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"- **{key}**: {value:.4f}")
                    else:
                        lines.append(f"- **{key}**: {str(value)[:100]}...")
            lines.append("")
    
    # Configuration section
    if results["configurations"]:
        lines.append("## Configurations")
        lines.append("")
        for config_info in results["configurations"]:
            config_file = Path(config_info["file"]).name
            lines.append(f"### {config_file}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(config_info["data"], indent=2)[:1000] + "...")
            lines.append("```")
            lines.append("")
    
    # File list section
    lines.append("## Processed Files")
    lines.append("")
    for file_path in results["metadata"]["collected_files"]:
        lines.append(f"- `{file_path}`")
    lines.append("")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Generated Markdown report: {output_file}")
    return output_file

def generate_html_report(results: Dict[str, Any], output_dir: Path, title: str) -> Path:
    """Generate HTML report with basic styling."""
    output_file = output_dir / "reproduction_report.html"
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
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
        h3 {{ color: #7f8c8d; }}
        .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-left: 4px solid #27ae60; }}
        .config {{ background-color: #f8f9fa; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; }}
        pre {{ background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .file-list {{ background-color: #fdf6e3; padding: 15px; border-radius: 5px; }}
        .timestamp {{ color: #7f8c8d; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Input Directory:</strong> {results['metadata']['input_directory']}</p>
        
        <div class="summary">
            <h2>📊 Summary</h2>
            <ul>
                <li><strong>Total Files Processed:</strong> {len(results['metadata']['collected_files'])}</li>
                <li><strong>Metrics Files:</strong> {len(results['metrics'])}</li>
                <li><strong>Sequential Analysis Files:</strong> {len(results['sequential_analysis'])}</li>
                <li><strong>Raw Result Files:</strong> {len(results['raw_results'])}</li>
                <li><strong>Configuration Files:</strong> {len(results['configurations'])}</li>
                <li><strong>Log Files:</strong> {len(results['logs'])}</li>
            </ul>
        </div>
"""
    
    # Add metrics section
    if results["metrics"]:
        html_content += "\n        <h2>📈 Metrics Results</h2>\n"
        for filename, metrics_data in results["metrics"].items():
            html_content += f"        <h3>{filename}</h3>\n"
            if isinstance(metrics_data, dict):
                for metric_name, value in metrics_data.items():
                    if isinstance(value, (int, float)):
                        html_content += f'        <div class="metric"><strong>{metric_name}:</strong> {value:.4f}</div>\n'
                    else:
                        html_content += f'        <div class="metric"><strong>{metric_name}:</strong> {value}</div>\n'
    
    # Add sequential analysis section
    if results["sequential_analysis"]:
        html_content += "\n        <h2>🔄 Sequential Analysis</h2>\n"
        for filename, seq_data in results["sequential_analysis"].items():
            html_content += f"        <h3>{filename}</h3>\n"
            if isinstance(seq_data, dict):
                for key, value in seq_data.items():
                    if isinstance(value, (int, float)):
                        html_content += f'        <div class="metric"><strong>{key}:</strong> {value:.4f}</div>\n'
                    else:
                        html_content += f'        <div class="metric"><strong>{key}:</strong> {str(value)[:100]}...</div>\n'
    
    # Add configurations section
    if results["configurations"]:
        html_content += "\n        <h2>⚙️ Configurations</h2>\n"
        for config_info in results["configurations"]:
            config_file = Path(config_info["file"]).name
            html_content += f"        <h3>{config_file}</h3>\n"
            html_content += '        <div class="config">\n'
            html_content += f'        <pre>{json.dumps(config_info["data"], indent=2)[:1000]}...</pre>\n'
            html_content += '        </div>\n'
    
    # Add file list
    html_content += "\n        <h2>📁 Processed Files</h2>\n"
    html_content += '        <div class="file-list">\n'
    html_content += "        <ul>\n"
    for file_path in results["metadata"]["collected_files"]:
        html_content += f"            <li><code>{file_path}</code></li>\n"
    html_content += "        </ul>\n"
    html_content += "        </div>\n"
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report: {output_file}")
    return output_file

def main():
    """Main function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {args.format} report(s) using {args.template} template")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Collect results
    results = collect_results(input_dir)
    
    # Generate reports based on format
    generated_files = []
    
    if args.format in ["json", "all"]:
        json_file = generate_json_report(results, output_dir, args.title)
        generated_files.append(json_file)
    
    if args.format in ["markdown", "all"]:
        md_file = generate_markdown_report(results, output_dir, args.title)
        generated_files.append(md_file)
    
    if args.format in ["html", "all"]:
        html_file = generate_html_report(results, output_dir, args.title)
        generated_files.append(html_file)
    
    if args.format == "pdf":
        logger.warning("PDF generation not yet implemented - use HTML for now")
    
    # Summary
    logger.info(f"Successfully generated {len(generated_files)} report(s):")
    for file_path in generated_files:
        logger.info(f"  - {file_path}")
    
    # Create index file for easy access
    index_file = output_dir / "index.html"
    index_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{args.title} - Index</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        .report-link {{ 
            display: block; 
            padding: 10px; 
            margin: 10px 0; 
            background-color: #f8f9fa; 
            border: 1px solid #dee2e6; 
            border-radius: 5px; 
            text-decoration: none; 
            color: #495057;
        }}
        .report-link:hover {{ background-color: #e9ecef; }}
    </style>
</head>
<body>
    <h1>{args.title}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>Available Reports:</h2>
"""
    
    for file_path in generated_files:
        filename = file_path.name
        index_content += f'    <a href="{filename}" class="report-link">📄 {filename}</a>\n'
    
    index_content += """
</body>
</html>
"""
    
    with open(index_file, 'w') as f:
        f.write(index_content)
    
    logger.info(f"Created index file: {index_file}")
    logger.info("Report generation completed successfully!")

if __name__ == "__main__":
    main()